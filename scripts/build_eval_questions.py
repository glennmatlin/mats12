"""Merge seed questions + worker candidates into frozen 100-question eval sets.

QC: parse worker JSON (tolerating code fences), drop near-duplicates
(normalized text), enforce length bounds, seeds first, cut to exactly 100.
Writes data/evals/<behavior>.json plus a provenance manifest. Fails loudly if
any behavior cannot reach 100 clean questions.
"""

import argparse
import json
import re
from pathlib import Path

from mats12.behaviors import BEHAVIORS

TARGET = 100
MIN_LEN, MAX_LEN = 12, 500


def normalize(q: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()


def parse_candidates(raw: str) -> list[str]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array found in worker output")
    try:
        items = json.loads(text[start : end + 1], strict=False)
    except json.JSONDecodeError:
        # Workers occasionally drop a closing quote on one line; recover the
        # well-formed lines rather than losing the whole batch.
        items = re.findall(r'^\s*"(.+?)",?\s*$', text, flags=re.MULTILINE)
    return [q.strip() for q in items if isinstance(q, str)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--generator", required=True, help="provenance: worker model used")
    args = parser.parse_args()

    seeds = json.loads(args.seeds.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for key in BEHAVIORS:
        candidates = parse_candidates((args.raw_dir / f"{key}.raw").read_text())
        merged: list[str] = []
        seen: set[str] = set()
        for q in list(seeds[key]) + candidates:
            norm = normalize(q)
            if not norm or norm in seen or not MIN_LEN <= len(q) <= MAX_LEN:
                continue
            seen.add(norm)
            merged.append(q)
        if len(merged) < TARGET:
            raise SystemExit(f"{key}: only {len(merged)} clean questions (< {TARGET})")
        final = merged[:TARGET]
        (args.out_dir / f"{key}.json").write_text(json.dumps(final, indent=1) + "\n")
        counts[key] = {"seeds": len(seeds[key]), "candidates": len(candidates), "final": TARGET}
        print(f"{key}: {len(candidates)} candidates -> {TARGET} final (seeds first)")

    manifest = {
        "date": "2026-08-13",
        "generator": args.generator,
        "seeds": "hand-written (data/evals/seeds.json); "
        "validate_feelings 1-3 verbatim from the post",
        "qc": "normalized dedupe, length 12-500 chars, seeds always retained, cut to 100",
        "refusal_severity": "LOW only by construction (see gen_eval_briefs.py rules)",
        "freeze": "sets are frozen at first eval use per PREREGISTRATION.md",
        "counts": counts,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("manifest written")


if __name__ == "__main__":
    main()
