"""Gate-clause-2 analysis: filtering effects for the retrain matrix.

Frozen rules (PREREGISTRATION.md @ e3eb75f):
- filtering effect := (S_filtered - S_base) / (S_SFT - S_base), paired means
  under this project's judge on the frozen question sets.
- "Filterable at 25%" := >= 50% of the SFT-minus-base delta removed relative
  to the equal-n random-removal baseline; "not filterable" := <= 20%; 20-50%
  indeterminate. Random baseline = mean of the two seeds per fraction
  (20260813, 20260814; fixed in the #8 sizing decision pre-unblinding).
  "Removed relative to the random baseline" is operationalized with the
  random-cell mean as the reference endpoint:
  removed = (S_random - S_filtered) / (S_random - S_base), which equals
  1 - effect_filtered / effect_random and nets out the delta lost by
  removing 25% of the data regardless of targeting.

CI: percentile bootstrap over questions (paired per question_id across all
conditions), 10,000 resamples, seed 20260816. Reads step1 base/sft scores
and gate2 retrain-cell scores; writes <out-dir>/gate2_analysis.json and
prints the verdict table.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TARGET_CELLS = {
    "refusal": {25: "refusal_probe_25", 10: "refusal_probe_10"},
    "both_sides": {25: "both_sides_bergson_25", 10: "both_sides_bergson_10"},
    # H2 tested units (#7, added 2026-08-26 after the unit set was fixed
    # pre-unblinding): 25% only - H2's effect metric is defined at 25%.
    # Listed AFTER the gate cells so their bootstrap draws are unchanged.
    "validate_feelings": {25: "validate_feelings_probe_25"},
    "bold_formatting": {25: "bold_formatting_probe_25"},
    "ethical_frameworks": {25: "ethical_frameworks_probe_25"},
}
RANDOM_CELLS = {
    25: ("random_25_s20260813", "random_25_s20260814"),
    10: ("random_10_s20260813", "random_10_s20260814"),
}


def load_scores(judging_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    """cond -> behavior -> question_id -> score."""
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for path in sorted(judging_dir.glob("scores_*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if not rows:
            continue
        behavior = rows[0]["behavior"]
        cond = path.stem.removeprefix("scores_").removesuffix(f"_{behavior}")
        out[cond][behavior] = {r["question_id"]: float(r["score"]) for r in rows}
    return out


def removed_fraction(base: np.ndarray, rand: np.ndarray, filt: np.ndarray) -> float:
    return float((rand.mean() - filt.mean()) / (rand.mean() - base.mean()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step1-judging", type=Path, default=Path("results/step1/judging"))
    parser.add_argument("--gate2-judging", type=Path, default=Path("results/gate2/judging"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/gate2"))
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    step1 = load_scores(args.step1_judging)
    gate2 = load_scores(args.gate2_judging)
    rng = np.random.default_rng(args.seed)

    report: dict = {
        "frozen_rule": "filterable >= 0.50 removed vs random baseline; "
        "not filterable <= 0.20; else indeterminate",
        "random_baseline": "mean of seeds 20260813, 20260814 per fraction",
        "seed": args.seed,
        "n_boot": args.n_boot,
        "targets": {},
        "filtering_effects_all_behaviors": {},
    }

    behaviors = sorted(step1["base"])
    for cell, conds in gate2.items():
        report["filtering_effects_all_behaviors"][cell] = {}
        for behavior in sorted(conds):
            qs = sorted(
                conds[behavior].keys()
                & step1["base"][behavior].keys()
                & step1["sft"][behavior].keys()
            )
            b = np.array([step1["base"][behavior][q] for q in qs])
            s = np.array([step1["sft"][behavior][q] for q in qs])
            f = np.array([conds[behavior][q] for q in qs])
            report["filtering_effects_all_behaviors"][cell][behavior] = {
                "n": len(qs),
                "mean": round(float(f.mean()), 3),
                "filtering_effect": round(float((f.mean() - b.mean()) / (s.mean() - b.mean())), 4),
            }

    for behavior, cells in TARGET_CELLS.items():
        report["targets"][behavior] = {}
        for pct, target_cell in cells.items():
            if target_cell not in gate2:
                print(f"note: {target_cell} not judged yet, skipping")
                continue
            r1, r2 = RANDOM_CELLS[pct]
            qs = sorted(
                step1["base"][behavior].keys()
                & step1["sft"][behavior].keys()
                & gate2[target_cell][behavior].keys()
                & gate2[r1][behavior].keys()
                & gate2[r2][behavior].keys()
            )
            b = np.array([step1["base"][behavior][q] for q in qs])
            s = np.array([step1["sft"][behavior][q] for q in qs])
            f = np.array([gate2[target_cell][behavior][q] for q in qs])
            r = np.array(
                [(gate2[r1][behavior][q] + gate2[r2][behavior][q]) / 2 for q in qs]
            )
            point = removed_fraction(b, r, f)
            idx = rng.integers(0, len(qs), size=(args.n_boot, len(qs)))
            boots = np.array([removed_fraction(b[i], r[i], f[i]) for i in idx])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            if point >= 0.50:
                verdict = "filterable"
            elif point <= 0.20:
                verdict = "not_filterable"
            else:
                verdict = "indeterminate"
            report["targets"][behavior][pct] = {
                "cell": target_cell,
                "n": len(qs),
                "means": {
                    "base": round(float(b.mean()), 3),
                    "sft": round(float(s.mean()), 3),
                    "random_baseline": round(float(r.mean()), 3),
                    "target_filtered": round(float(f.mean()), 3),
                },
                "effect_filtered": round(
                    float((f.mean() - b.mean()) / (s.mean() - b.mean())), 4
                ),
                "effect_random": round(float((r.mean() - b.mean()) / (s.mean() - b.mean())), 4),
                "removed_vs_random": round(point, 4),
                "ci95": [round(float(lo), 4), round(float(hi), 4)],
                "verdict": verdict,
            }
            print(
                f"{behavior} @ {pct}%: removed {point:.1%} vs random "
                f"[{lo:.1%}, {hi:.1%}] -> {verdict.upper()}"
            )

    for cell in sorted(gate2):
        eff = report["filtering_effects_all_behaviors"][cell]
        missing = [x for x in behaviors if x not in eff]
        if missing:
            print(f"note: {cell} missing behaviors {missing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "gate2_analysis.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
