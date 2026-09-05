import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
_SPEC = importlib.util.spec_from_file_location("summarize_r1", SCRIPTS_DIR / "summarize_r1.py")
assert _SPEC is not None and _SPEC.loader is not None
summarize_r1 = importlib.util.module_from_spec(_SPEC)
sys.modules["summarize_r1"] = summarize_r1
_SPEC.loader.exec_module(summarize_r1)


EXPECTED_INSTALLATION = {
    "both_sides": {
        "n": 100,
        "base_mean": -0.69,
        "full_ft_mean": 3.64,
        "paired_delta": 4.33,
        "ci95": [3.56, 5.07],
        "ci_excludes_zero": True,
    },
    "refusal": {
        "n": 100,
        "base_mean": -1.04,
        "full_ft_mean": 3.49,
        "paired_delta": 4.53,
        "ci95": [3.87, 5.175],
        "ci_excludes_zero": True,
    },
    "validate_feelings": {
        "n": 100,
        "base_mean": -2.965,
        "full_ft_mean": 3.005,
        "paired_delta": 5.97,
        "ci95": [5.28, 6.615],
        "ci_excludes_zero": True,
    },
}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_behavior_inputs(
    repo_root: Path, behavior: str, mismatched_ids: bool, score_mismatch: bool
) -> list[dict[str, Any]]:
    base_qids = [f"{behavior}-0", f"{behavior}-1"]
    full_ft_qids = (
        [f"{behavior}-0", f"{behavior}-other"]
        if mismatched_ids and behavior == "refusal"
        else base_qids
    )
    base_dir = repo_root / "results" / "step1" / "judging"
    full_ft_dir = repo_root / "results" / "r1" / "judging"
    gens_dir = repo_root / "results" / "r1" / "gens"
    write_jsonl(
        base_dir / f"scores_base_{behavior}.jsonl",
        [{"question_id": qid, "score": 0} for qid in base_qids],
    )
    write_jsonl(
        full_ft_dir / f"scores_r1_full_ft_{behavior}.jsonl",
        [
            {
                "question_id": qid,
                "score": 2 if score_mismatch and qid == "both_sides-0" else 1,
                "generation": "ok",
                "judge": "claude-cli",
            }
            for qid in full_ft_qids
        ],
    )
    write_jsonl(
        gens_dir / f"{behavior}.jsonl",
        [
            {"question_id": qid, "completion": "ok", "think_truncated": False}
            for qid in full_ft_qids
        ],
    )
    completion_hash = hashlib.sha256(b"ok").hexdigest()[:12]
    return [
        {
            "cond": "r1_full_ft",
            "behavior": behavior,
            "question_id": qid,
            "provider": "claude-cli",
            "completion_sha": completion_hash,
            "score": 1,
            "error": None,
        }
        for qid in base_qids
    ]


def write_minimal_inputs(
    repo_root: Path,
    *,
    mismatched_ids: bool = False,
    stale_judgment: bool = False,
    duplicate_judgment: bool = False,
    score_mismatch: bool = False,
    missing_error: bool = False,
) -> None:
    directories = (
        repo_root / "results" / "step1" / "judging",
        repo_root / "results" / "r1" / "judging",
        repo_root / "results" / "r1" / "gens",
        repo_root / "scripts",
    )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    for script in ("summarize_r1.py", "summarize_step1.py", "judge_sharded.py"):
        (repo_root / "scripts" / script).write_text(f"{script}\n")
    judgments = [
        row
        for behavior in summarize_r1.BEHAVIORS
        for row in write_behavior_inputs(repo_root, behavior, mismatched_ids, score_mismatch)
    ]
    if stale_judgment:
        judgments[0]["completion_sha"] = "000000000000"
    if missing_error:
        judgments[0].pop("error")
    if duplicate_judgment:
        judgments.append(judgments[0].copy())
    write_jsonl(repo_root / "results" / "r1" / "judging" / "raw_judgments.jsonl", judgments)


@pytest.fixture(scope="module")
def committed_receipt() -> dict[str, Any]:
    return summarize_r1.build_installation_receipt(REPO_ROOT, seed=20260816, n_boot=10_000)


@pytest.mark.parametrize(("behavior", "expected"), EXPECTED_INSTALLATION.items())
def test_installation_stats_match_committed_r1_rows(
    committed_receipt: dict[str, Any], behavior: str, expected: dict[str, object]
) -> None:
    assert committed_receipt["behaviors"][behavior] == expected


def test_receipt_records_bootstrap_and_completeness(committed_receipt: dict[str, Any]) -> None:
    assert committed_receipt["bootstrap"] == {
        "method": "paired question-level percentile bootstrap",
        "rng": "numpy.random.default_rng",
        "seed": 20260816,
        "resamples": 10_000,
        "percentiles": [2.5, 97.5],
        "behavior_order": ["both_sides", "refusal", "validate_feelings"],
    }
    assert committed_receipt["completeness"] == {
        "generation_rows": 300,
        "nonempty_generations": 300,
        "think_truncated_generations": 0,
        "score_rows": 300,
        "judgment_rows": 300,
        "unique_judgments": 300,
        "content_bound_judgments": 300,
        "judgment_errors": 0,
        "question_id_sets_match": True,
    }


def test_receipt_binds_inputs_and_producers(committed_receipt: dict[str, Any]) -> None:
    assert committed_receipt["inputs"]["full_ft_scores"]["both_sides"]["sha256"] == (
        "ca2d71e963ec4dab1db9597b221bd65267313aa942d5e6e791dce9502b042d76"
    )
    assert committed_receipt["inputs"]["base_scores"]["validate_feelings"]["sha256"] == (
        "3cbcbc552bf546894a443ddc711bbd8fe220a151089090ef3b1285c645efd7a4"
    )
    assert [item["path"] for item in committed_receipt["producer"]["files"]] == [
        "scripts/summarize_r1.py",
        "scripts/summarize_step1.py",
        "scripts/judge_sharded.py",
    ]
    assert all(len(item["sha256"]) == 64 for item in committed_receipt["producer"]["files"])


def test_receipt_rejects_mismatched_question_ids(tmp_path: Path) -> None:
    write_minimal_inputs(tmp_path, mismatched_ids=True)

    with pytest.raises(ValueError, match="refusal.*question ids do not match"):
        summarize_r1.build_installation_receipt(tmp_path, seed=1, n_boot=10)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"stale_judgment": True}, "both_sides-0.*does not match generation"),
        ({"duplicate_judgment": True}, "duplicate judgment"),
        ({"score_mismatch": True}, "both_sides-0.*does not match judgment score"),
        ({"missing_error": True}, "judgment row missing fields"),
        ({}, "behavior counts must each be 100"),
    ],
)
def test_receipt_fails_closed_on_incomplete_or_inconsistent_evidence(
    tmp_path: Path, options: dict[str, bool], message: str
) -> None:
    write_minimal_inputs(tmp_path, **options)

    with pytest.raises(ValueError, match=message):
        summarize_r1.build_installation_receipt(tmp_path, seed=1, n_boot=10)
