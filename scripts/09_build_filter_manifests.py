"""Build filter manifests for the gate-clause-2 retrain matrix (#8 sizing).

Matrix fixed 2026-08-16 (owner decision, pre-unblinding): fractions 25% and
10%; scored cells use each behavior's BINDING scorer (refusal = probe;
both_sides = Bergson gradient TDA per the trigger-2 escalation, built
separately when its scores exist); random baselines are behavior-agnostic
(random removal does not depend on the target behavior; each baseline
retrain is evaluated on all behaviors), TWO seeds per fraction - 20260813
and 20260814 - whose mean is the baseline per the frozen decision rule.

Each manifest is {"cell", "drop_ids": [...]} consumed by
02_train_speedrun.py --filter-manifest.
"""

import argparse
import json
from pathlib import Path

import numpy as np

FRACTIONS = (0.25, 0.10)
RANDOM_SEEDS = (20260813, 20260814)
PROBE_BEHAVIORS = ("refusal",)
# H2 tested units added 2026-08-25 (#7, unit set fixed pre-unblinding): each
# needs its own-direction 25% cell. All three hold trigger 2, so they stay on
# the probe scorer. 25% only - H2's effect metric is defined at 25% removal.


def write_manifest(out_dir: Path, cell: str, drop_ids: list[str]) -> None:
    (out_dir / f"{cell}.json").write_text(
        json.dumps({"cell": cell, "n_dropped": len(drop_ids), "drop_ids": drop_ids}) + "\n"
    )
    print(f"{cell}: {len(drop_ids)} dropped")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-scores",
        type=Path,
        default=Path("results/attribution/corpus-scores/probe_scores.npz"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/attribution/manifests"))
    parser.add_argument(
        "--behaviors", default=",".join(PROBE_BEHAVIORS), help="probe-scored cells to build"
    )
    parser.add_argument("--fractions", default=",".join(str(f) for f in FRACTIONS))
    parser.add_argument(
        "--skip-random", action="store_true", help="leave the existing random baselines untouched"
    )
    args = parser.parse_args()
    behaviors = tuple(b.strip() for b in args.behaviors.split(",") if b.strip())
    fractions = tuple(float(f) for f in args.fractions.split(",") if f.strip())

    data = np.load(args.probe_scores)
    ids = data["ids"]
    n = len(ids)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for behavior in behaviors:
        scores = data[f"scores_{behavior}"]
        order = np.argsort(-scores)
        for frac in fractions:
            k = int(frac * n)
            write_manifest(
                args.out_dir,
                f"{behavior}_probe_{int(frac * 100)}",
                [str(x) for x in ids[order[:k]]],
            )

    for frac in () if args.skip_random else fractions:
        k = int(frac * n)
        for seed in RANDOM_SEEDS:
            rng = np.random.default_rng(seed)
            pick = rng.choice(n, size=k, replace=False)
            write_manifest(
                args.out_dir,
                f"random_{int(frac * 100)}_s{seed}",
                [str(x) for x in ids[pick]],
            )


if __name__ == "__main__":
    main()
