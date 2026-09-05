"""Part C qualitative examples: seeded random draw, committed pre-selection.

The admissions doc requires randomly selected (not cherry-picked) judged
generations right after the exec summary. This script IS the verifiability:
the seed and protocol are committed before any example text is viewed.
Protocol: from the Step-1 base and sft conditions, draw 3 behaviors uniformly
(replication triple always included), then 1 question per (behavior,
condition), n = 6 examples total, seed 20260816.
"""

import argparse
import json
import random
from pathlib import Path

from mats12.behaviors import REPLICATION_TRIPLE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judging", type=Path, default=Path("results/step1/judging"))
    parser.add_argument("--gens-root", type=Path, default=Path("results/gate2"))
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--out", type=Path, default=Path("docs/writeup/qualitative_examples.md"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    behaviors = sorted(REPLICATION_TRIPLE)
    lines = [
        "# Randomly selected qualitative examples (Part C)",
        "",
        f"Protocol committed pre-draw (`scripts/21_qualitative_sample.py`, seed "
        f"{args.seed}): the replication triple, one random question per "
        "(behavior, condition) from the Step-1 judged scores, base and sft "
        "conditions. Scores are the single-Claude-judge values used everywhere "
        "else; generations quoted verbatim, truncated at 900 characters.",
        "",
    ]
    for behavior in behaviors:
        for cond in ("base", "sft"):
            rows = [
                json.loads(line)
                for line in (args.judging / f"scores_{cond}_{behavior}.jsonl")
                .read_text()
                .splitlines()
                if line
            ]
            r = rng.choice(rows)
            gen = r["generation"]
            clipped = gen[:900] + (f"\n... [{len(gen) - 900} more chars]" if len(gen) > 900 else "")
            lines += [
                f"## {behavior} / {cond} / {r['question_id']} - judge score {r['score']:+.1f}",
                "",
                "```",
                clipped,
                "```",
                "",
            ]
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
