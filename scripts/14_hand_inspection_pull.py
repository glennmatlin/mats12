"""Random top-doc hand-inspection pull (ticket #7, admissions data-hygiene rule).

Protocol (fixing the "hand-inspection protocol detail" left unspecified in the
wayfinder map, logged in LOGBOOK 2026-08-25): for each cell, sample n=10
documents uniformly WITHOUT replacement from the top-1000 of that cell's
ranking (k matches P1a), seed 20260816. Cells = the seven probe rankings plus
both_sides under Bergson (its binding scorer per the trigger-2 escalation).
A 10-doc uniform-corpus baseline is drawn for contrast. Selection is committed
before any document text is read.

Two phases so the text fetch can happen on PACE between them:
  --emit-ids   write sample_manifest.json + the deduped id list to fetch
  --render     merge fetched texts (jsonl: {id, dataset_source, messages})
               into INSPECTION.md + docs.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np

TOP_K = 1000
N_PER_CELL = 10
EXCERPT_CHARS = 1500


def load_bergson(run_dir: Path) -> tuple[list[str], np.ndarray]:
    info = json.loads((run_dir / "info.json").read_text())
    dtype = np.dtype(
        {
            "names": info["dtype"]["names"],
            "formats": info["dtype"]["formats"],
            "offsets": info["dtype"]["offsets"],
            "itemsize": info["dtype"]["itemsize"],
        }
    )
    records = np.fromfile(run_dir / "scores.bin", dtype=dtype)
    ids = json.loads((run_dir.parent / "corpus_ids.json").read_text())
    return ids, records["score_0"].astype(np.float64)


def sample_cell(ids: list[str], scores: np.ndarray, rng: np.random.Generator) -> list[dict]:
    order = np.argsort(-scores)
    pick = rng.choice(TOP_K, size=N_PER_CELL, replace=False)
    return [
        {"id": str(ids[order[r]]), "rank": int(r) + 1, "score": float(scores[order[r]])}
        for r in sorted(pick.tolist())
    ]


def emit_ids(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    probe = np.load(args.probe_scores)
    probe_ids = [str(x) for x in probe["ids"]]
    behaviors = sorted(k.removeprefix("scores_") for k in probe.files if k.startswith("scores_"))

    cells: dict[str, list[dict]] = {}
    for b in behaviors:  # sorted order pins the rng stream
        cells[f"{b}_probe"] = sample_cell(probe_ids, probe[f"scores_{b}"].astype(np.float64), rng)
    bergson_ids, bergson_scores = load_bergson(args.bergson_dir / "both_sides")
    cells["both_sides_bergson"] = sample_cell(bergson_ids, bergson_scores, rng)
    baseline = rng.choice(len(probe_ids), size=N_PER_CELL, replace=False)
    cells["corpus_baseline"] = [
        {"id": probe_ids[i], "rank": None, "score": None} for i in sorted(baseline.tolist())
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": (
            f"n={N_PER_CELL} uniform without replacement from top-{TOP_K} per cell; "
            f"seed {args.seed}; cells drawn in sorted-behavior order then "
            "both_sides_bergson then corpus_baseline; committed before any text read"
        ),
        "seed": args.seed,
        "cells": cells,
    }
    (args.out_dir / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    wanted = sorted({d["id"] for cell in cells.values() for d in cell})
    (args.out_dir / "wanted_ids.txt").write_text("\n".join(wanted) + "\n")
    print(f"{len(wanted)} unique ids across {len(cells)} cells -> {args.out_dir}")


def render_messages(messages: list[dict]) -> str:
    return "\n\n".join(f"[{m.get('role', '?')}] {m.get('content', '')}" for m in messages)


def render(args: argparse.Namespace) -> None:
    manifest = json.loads((args.out_dir / "sample_manifest.json").read_text())
    docs = {}
    with args.docs.open() as fh:
        for line in fh:
            d = json.loads(line)
            docs[d["id"]] = d
    (args.out_dir / "docs.jsonl").write_text(
        "".join(json.dumps(docs[i]) + "\n" for i in sorted(docs))
    )

    lines = [
        "# Hand-inspection sample (ticket #7)",
        "",
        f"Protocol: {manifest['protocol']}.",
        "",
        "Excerpts are the first "
        f"{EXCERPT_CHARS} characters of the role-tagged conversation; full texts "
        "in `docs.jsonl`. Inspection notes go under each cell heading.",
        "",
    ]
    for cell, rows in manifest["cells"].items():
        lines += [f"## {cell}", ""]
        for row in rows:
            doc = docs.get(row["id"])
            head = f"### rank {row['rank']}" if row["rank"] else "### baseline"
            score = f" — score {row['score']:.4g}" if row["score"] is not None else ""
            lines += [f"{head}{score} — `{row['id']}`", ""]
            if doc is None:
                lines += ["⚠️ text not fetched", ""]
                continue
            text = render_messages(doc["messages"])
            clipped = text[:EXCERPT_CHARS]
            suffix = (
                f"\n… [{len(text) - EXCERPT_CHARS} more chars]" if len(text) > EXCERPT_CHARS else ""
            )
            lines += [
                f"source: `{doc.get('dataset_source', '?')}` — {len(text)} chars",
                "",
                "```",
                clipped + suffix,
                "```",
                "",
                "**Notes:** _pending_",
                "",
            ]
    (args.out_dir / "INSPECTION.md").write_text("\n".join(lines))
    missing = [
        row["id"] for rows in manifest["cells"].values() for row in rows if row["id"] not in docs
    ]
    print(f"INSPECTION.md written; {len(missing)} missing texts {missing[:3]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-ids", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--probe-scores",
        type=Path,
        default=Path("results/attribution/corpus-scores/probe_scores.npz"),
    )
    parser.add_argument(
        "--bergson-dir", type=Path, default=Path("results/attribution/bergson-scores")
    )
    parser.add_argument("--docs", type=Path, help="fetched texts jsonl (render phase)")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--out-dir", type=Path, default=Path("results/h1h2/hand_inspection"))
    args = parser.parse_args()
    if args.emit_ids:
        emit_ids(args)
    if args.render:
        render(args)
    if not (args.emit_ids or args.render):
        parser.error("pass --emit-ids and/or --render")


if __name__ == "__main__":
    main()
