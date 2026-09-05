"""Regression tests for the content-bound judging ledger (code review 2026-08-16).

The red case these encode: a generation file is rewritten with new text after
its items were judged; the old ledger key (cond, question_id, provider) would
skip the items and silently pair stale scores with the new bytes.
"""

import importlib.util
import json
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "judge_sharded", Path(__file__).parent.parent / "scripts" / "judge_sharded.py"
)
assert _SPEC is not None and _SPEC.loader is not None
judge_sharded = importlib.util.module_from_spec(_SPEC)
sys.modules["judge_sharded"] = judge_sharded
_SPEC.loader.exec_module(judge_sharded)


def write_gen(gen_dir: Path, behavior: str, completion: str) -> None:
    gen_dir.mkdir(parents=True, exist_ok=True)
    (gen_dir / f"{behavior}.jsonl").write_text(
        json.dumps({"question_id": "q1", "question": "How do I feel?", "completion": completion})
        + "\n"
    )


def counting_client(calls: list):
    def client(prompt: str) -> str:
        calls.append(prompt)
        return "<reasoning>ok</reasoning><score>2.0</score>"

    return client


def test_rewritten_generation_is_rejudged_not_inherited(tmp_path):
    gen_dir = tmp_path / "gens"
    write_gen(gen_dir, "validate_feelings", "completion A")
    ledger = judge_sharded.Ledger(tmp_path / "raw_judgments.jsonl")
    calls: list = []
    clients = {"claude-cli": counting_client(calls)}

    items = judge_sharded.load_items({"base": gen_dir})
    work = [(it, "claude-cli") for it in items]
    judge_sharded.run_batch(work, ledger, clients, threads=1)
    assert len(calls) == 1

    # Resume with identical bytes: nothing re-judged.
    judge_sharded.run_batch(work, ledger, clients, threads=1)
    assert len(calls) == 1

    # Rewrite the generation file; the item MUST be judged again, and the
    # stale row must not be reachable under the new item's key.
    write_gen(gen_dir, "validate_feelings", "completion B, entirely different")
    items_b = judge_sharded.load_items({"base": gen_dir})
    old_key = judge_sharded.item_key(items[0], "claude-cli")
    new_key = judge_sharded.item_key(items_b[0], "claude-cli")
    assert old_key != new_key
    judge_sharded.run_batch([(it, "claude-cli") for it in items_b], ledger, clients, threads=1)
    assert len(calls) == 2
    assert ledger.rows[new_key]["completion_sha"] == items_b[0]["sha"]


def test_error_rows_are_retried_on_rerun(tmp_path):
    gen_dir = tmp_path / "gens"
    write_gen(gen_dir, "validate_feelings", "completion A")
    ledger = judge_sharded.Ledger(tmp_path / "raw_judgments.jsonl")
    items = judge_sharded.load_items({"base": gen_dir})
    ledger.record(
        {
            "cond": "base",
            "behavior": "validate_feelings",
            "question_id": "q1",
            "provider": "claude-cli",
            "completion_sha": items[0]["sha"],
            "score": None,
            "reasoning": "",
            "error": "timeout",
        }
    )
    calls: list = []
    judge_sharded.run_batch(
        [(it, "claude-cli") for it in items], ledger, {"claude-cli": counting_client(calls)}, 1
    )
    assert len(calls) == 1  # error row retried
    key = judge_sharded.item_key(items[0], "claude-cli")
    assert ledger.rows[key]["score"] == 2.0  # latest row wins in memory

    # And on replay from disk (append-only file, last line per key wins).
    reloaded = judge_sharded.Ledger(tmp_path / "raw_judgments.jsonl")
    assert reloaded.rows[key]["score"] == 2.0


def test_legacy_rows_without_sha_are_not_reused(tmp_path):
    gen_dir = tmp_path / "gens"
    write_gen(gen_dir, "validate_feelings", "completion A")
    (tmp_path / "raw_judgments.jsonl").write_text(
        json.dumps(
            {
                "cond": "base",
                "behavior": "validate_feelings",
                "question_id": "q1",
                "provider": "claude-cli",
                "score": -3.0,
                "reasoning": "judged before hash binding",
                "error": None,
            }
        )
        + "\n"
    )
    ledger = judge_sharded.Ledger(tmp_path / "raw_judgments.jsonl")
    items = judge_sharded.load_items({"base": gen_dir})
    calls: list = []
    judge_sharded.run_batch(
        [(it, "claude-cli") for it in items], ledger, {"claude-cli": counting_client(calls)}, 1
    )
    assert len(calls) == 1  # legacy unhashed row is not trusted for these bytes
