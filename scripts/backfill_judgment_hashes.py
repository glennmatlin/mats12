"""One-time migration (2026-08-16): bind existing judgments to their bytes.

Adds completion_sha (sha256[:12] of the judged completion) to every row of
results/step1/judging/raw_judgments.jsonl by joining (cond, behavior,
question_id) against the generation dirs the scores were produced from.
Run while gens and ledger still match exactly (verified 2026-08-16 during
code review); aborts if any row has no matching generation. Idempotent.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from judge_sharded import completion_sha  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger", type=Path, default=Path("results/step1/judging/raw_judgments.jsonl")
    )
    parser.add_argument(
        "--gen-dir",
        action="append",
        metavar="COND=PATH",
        default=None,
        help="default: base=results/step1/gens-base sft=results/step1/gens-sft",
    )
    args = parser.parse_args()
    specs = args.gen_dir or ["base=results/step1/gens-base", "sft=results/step1/gens-sft"]

    shas: dict[tuple[str, str, str], str] = {}
    for spec in specs:
        cond, _, path = spec.partition("=")
        for f in sorted(Path(path).glob("*.jsonl")):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                shas[(cond, f.stem, r["question_id"])] = completion_sha(r["completion"])

    rows = [json.loads(line) for line in args.ledger.read_text().splitlines()]
    added = already = 0
    for r in rows:
        if "completion_sha" in r:
            already += 1
            continue
        key = (r["cond"], r["behavior"], r["question_id"])
        if key not in shas:
            raise SystemExit(f"no generation found for ledger row {key}; aborting, nothing written")
        r["completion_sha"] = shas[key]
        added += 1

    tmp = args.ledger.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(args.ledger)
    print(f"backfilled {added} rows ({already} already had completion_sha, {len(rows)} total)")


if __name__ == "__main__":
    main()
