"""H3 verdicts: decontaminated refilter vs matched-method baseline.

Frozen rules (PREREGISTRATION @ e3eb75f + interpretation LOGBOOK 2026-08-27):
- P3a (validate_feelings): decontaminated cell IMPROVES filtering iff the
  baseline-minus-decontam judged-point difference on the target behavior is
  >= +1.0 with the 95% bootstrap CI excluding 0. (Improved filtering = LOWER
  behavior score after retraining, so improvement = S_baseline - S_decontam.)
- P3b (both_sides): does NOT improve, |difference| < 1.0 as an explicit
  equivalence bound (single-seed limitation disclosed).
- Pairing per question_id; percentile bootstrap, 10,000 resamples, seed
  20260816 (the gate2 convention).
"""

import argparse
import json
from pathlib import Path

import numpy as np

PAIRS = {
    "validate_feelings": ("validate_feelings_decontam_25", "validate_feelings_probe_25", "P3a"),
    "both_sides": ("both_sides_decontam_25", "both_sides_probe_25", "P3b"),
}


def scores_of(judging: Path, cell: str, behavior: str) -> dict[str, float]:
    rows = [
        json.loads(line)
        for line in (judging / f"scores_{cell}_{behavior}.jsonl").read_text().splitlines()
        if line
    ]
    return {r["question_id"]: float(r["score"]) for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judging", type=Path, default=Path("results/gate2/judging"))
    parser.add_argument("--out", type=Path, default=Path("results/h3/h3_analysis.json"))
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    report: dict = {
        "rules": (
            "P3a improves := baseline-minus-decontam >= +1.0, 95% CI excluding 0; "
            "P3b not-improve := |difference| < 1.0 equivalence bound; paired per "
            "question, 10k bootstrap, seed 20260816"
        ),
        "verdicts": {},
    }
    for behavior, (decontam, baseline, tag) in PAIRS.items():
        s_dec = scores_of(args.judging, decontam, behavior)
        s_base = scores_of(args.judging, baseline, behavior)
        qs = sorted(s_dec.keys() & s_base.keys())
        d = np.array([s_base[q] - s_dec[q] for q in qs])
        idx = rng.integers(0, len(qs), size=(args.n_boot, len(qs)))
        boots = d[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        point = float(d.mean())
        if tag == "P3a":
            holds = point >= 1.0 and lo > 0
            verdict = "improves" if holds else "does_not_improve"
        else:
            holds = abs(point) < 1.0
            verdict = "equivalent_no_improvement" if holds else "difference_exceeds_bound"
        report["verdicts"][behavior] = {
            "prediction": tag,
            "cells": {"decontam": decontam, "matched_baseline": baseline},
            "n": len(qs),
            "mean_decontam": round(float(np.mean([s_dec[q] for q in qs])), 3),
            "mean_baseline": round(float(np.mean([s_base[q] for q in qs])), 3),
            "improvement_baseline_minus_decontam": round(point, 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "prediction_holds": holds,
            "verdict": verdict,
        }
        print(
            f"{behavior} ({tag}): improvement {point:+.3f} CI [{lo:+.3f}, {hi:+.3f}] -> {verdict}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
