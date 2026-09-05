"""Regenerate the Step-1 judging summary and agreement report from raw artifacts.

Sources: results/step1/judging/scores_{cond}_{behavior}.jsonl (per-behavior
judge scores) and raw_judgments.jsonl (full ledger, including the phase-A
multi-provider revalidation rows). Outputs: summary.md (paired delta table
with bootstrap CIs) and agreement_report.json (pairwise judge agreement).
Deterministic given --seed, so every number is re-derivable per AGENTS.md.
"""

import argparse
import itertools
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

log = logging.getLogger("summarize_step1")

CONDITIONS = ("base", "sft")
PRIMARY_JUDGE = "claude-cli"
AGREEMENT_THRESHOLD = 1.0


def load_scores(judging_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    """behavior -> cond -> question_id -> score, from scores_{cond}_{behavior}.jsonl."""
    scores: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for path in sorted(judging_dir.glob("scores_*.jsonl")):
        cond, behavior = path.stem.removeprefix("scores_").split("_", 1)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        scores[behavior][cond] = {r["question_id"]: float(r["score"]) for r in rows}
    return scores


def paired_delta_ci(
    base: dict[str, float],
    sft: dict[str, float],
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, float | int | bool]:
    shared = sorted(base.keys() & sft.keys())
    if len(shared) != len(base) or len(shared) != len(sft):
        raise ValueError(
            f"unpaired question ids: base={len(base)} sft={len(sft)} shared={len(shared)}"
        )
    deltas = np.array([sft[q] - base[q] for q in shared])
    boot = rng.choice(deltas, size=(n_boot, len(deltas)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": len(shared),
        "base_mean": float(np.mean([base[q] for q in shared])),
        "sft_mean": float(np.mean([sft[q] for q in shared])),
        "delta_mean": float(deltas.mean()),
        "delta_se": float(deltas.std(ddof=1) / np.sqrt(len(deltas))),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


def pairwise_agreement(raw_path: Path) -> dict:
    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    by_item: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in rows:
        by_item[(r["cond"], r["behavior"], r["question_id"])][r["provider"]] = float(r["score"])
    multi = {k: v for k, v in by_item.items() if len(v) > 1}
    pairs: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for scores in multi.values():
        for a, b in itertools.combinations(sorted(scores), 2):
            pairs[(a, b)].append((scores[a], scores[b]))
    pairwise = {
        f"{a}|{b}": {
            "n_items": len(xs),
            "within_1": round(sum(abs(x - y) <= AGREEMENT_THRESHOLD for x, y in xs) / len(xs), 4),
            "mean_abs_diff": round(sum(abs(x - y) for x, y in xs) / len(xs), 4),
        }
        for (a, b), xs in sorted(pairs.items())
    }
    return {
        "mode": "claude-only",
        "primary_judge": PRIMARY_JUDGE,
        "providers": sorted({r["provider"] for r in rows}),
        "ledger_rows": len(rows),
        "ledger_errors": sum(1 for r in rows if r["error"]),
        "phase_a_revalidation_items": len(multi),
        "within_1_threshold": AGREEMENT_THRESHOLD,
        "pairwise_within_1": pairwise,
        "fallback_decision": (
            "Phase-A revalidation across claude/grok/codex collapsed vs the n=3 "
            "smoke (within-1 0.54-0.60, mean |diff| 1.26-2.05), so the sharded "
            "protocol auto-fell back to the single Claude judge - the anchor "
            "post's own setup. Ledgered in LOGBOOK 2026-08-14 03:05."
        ),
    }


def render_summary(stats: dict[str, dict], agreement: dict) -> str:
    lines = [
        "# Step 1 judging summary",
        "",
        "Judge: single Claude judge (`claude-cli`), behavior-presence rubric scored",
        "-5.0..+5.0 (rubric anchors per behavior in `src/mats12/behaviors.py`;",
        "template in `src/mats12/judge.py`, question-visible, change ledgered",
        "pre-scoring). Multi-judge sharding was abandoned after the phase-A",
        "revalidation collapse recorded in `agreement_report.json`; claude-only",
        "matches the anchor post's own setup.",
        "",
        "Deltas are paired per `question_id` (base and sft score identical question",
        "sets). CI: percentile bootstrap over questions, 10,000 resamples, seed",
        "20260816. Regenerate: `uv run python scripts/summarize_step1.py`.",
        "",
        "| behavior | base | sft | paired delta | 95% CI | excludes 0 |",
        "|---|---|---|---|---|---|",
    ]
    for behavior, s in sorted(stats.items()):
        lines.append(
            f"| {behavior} | {s['base_mean']:+.2f} (n={s['n']}) "
            f"| {s['sft_mean']:+.2f} (n={s['n']}) | {s['delta_mean']:+.2f} "
            f"| [{s['ci95_lo']:+.2f}, {s['ci95_hi']:+.2f}] "
            f"| {'yes' if s['ci_excludes_zero'] else 'no'} |"
        )
    lines += [
        "",
        f"Judge agreement (phase-A revalidation, n={agreement['phase_a_revalidation_items']}"
        " items, within-1.0 rate / mean |diff|):",
        "",
    ]
    for pair, a in agreement["pairwise_within_1"].items():
        lines.append(f"- {pair}: {a['within_1']:.2f} / {a['mean_abs_diff']:.2f}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judging-dir", type=Path, default=Path("results/step1/judging"))
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rng = np.random.default_rng(args.seed)
    scores = load_scores(args.judging_dir)
    stats = {}
    for behavior in sorted(scores):
        conds = scores[behavior]
        missing = [c for c in CONDITIONS if c not in conds]
        if missing:
            raise ValueError(f"behavior '{behavior}' missing conditions: {missing}")
        stats[behavior] = paired_delta_ci(conds["base"], conds["sft"], rng, args.n_boot)

    agreement = pairwise_agreement(args.judging_dir / "raw_judgments.jsonl")
    agreement_path = args.judging_dir / "agreement_report.json"
    agreement_path.write_text(json.dumps(agreement, indent=2) + "\n")
    log.info("wrote %s", agreement_path)

    summary_path = args.judging_dir / "summary.md"
    summary_path.write_text(render_summary(stats, agreement))
    log.info("wrote %s", summary_path)
    for behavior, s in sorted(stats.items()):
        log.info(
            "%s: delta %+.2f [%+.2f, %+.2f]",
            behavior,
            s["delta_mean"],
            s["ci95_lo"],
            s["ci95_hi"],
        )


if __name__ == "__main__":
    main()
