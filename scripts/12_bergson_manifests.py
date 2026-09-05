"""Build both_sides_bergson_{25,10} filter manifests from trackstar scores.

Reads the trackstar run's scores.bin (structured records {score_0: float32,
written_0: bool}, one per corpus doc, in corpus.jsonl row order) and drops
the top fraction by score, mirroring 09_build_filter_manifests.py. Verifies
every record was written and that counts match the corpus before trusting
the order-based id join.
"""

import argparse
import json
from pathlib import Path

import numpy as np

FRACTIONS = (0.25, 0.10)


def load_scores(run_dir: Path) -> np.ndarray:
    if (run_dir / "scores").is_dir():
        run_dir = run_dir / "scores"
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
    if len(records) != info["num_items"]:
        raise SystemExit(f"record count {len(records)} != num_items {info['num_items']}")
    if not records["written_0"].all():
        raise SystemExit(f"{(~records['written_0']).sum()} unwritten score records")
    return records["score_0"].astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="trackstar_<behavior> dir")
    parser.add_argument("--corpus", type=Path, required=True, help="corpus.jsonl (id order)")
    parser.add_argument("--behavior", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/attribution/manifests"))
    args = parser.parse_args()

    ids = [json.loads(line)["id"] for line in args.corpus.read_text().splitlines() if line]
    scores = load_scores(args.run_dir)
    if len(ids) != len(scores):
        raise SystemExit(f"corpus {len(ids)} docs != {len(scores)} scores")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-scores)
    for frac in FRACTIONS:
        k = int(frac * len(ids))
        cell = f"{args.behavior}_bergson_{int(frac * 100)}"
        drop = [ids[i] for i in order[:k]]
        (args.out_dir / f"{cell}.json").write_text(
            json.dumps({"cell": cell, "n_dropped": len(drop), "drop_ids": drop}) + "\n"
        )
        print(f"{cell}: {len(drop)} dropped")


if __name__ == "__main__":
    main()
