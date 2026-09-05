"""H4 verdict: persona-axis concentration over the WebOrganizer bins vs the null.

(577 distinct bin ids in the manifest; the docs said 576, the artifact is authoritative.)
Implements exactly the readout pinned in LOGBOOK 2026-08-28 before any
persona-axis aggregate was viewed: per direction, the top-decile doc mass
across bins m_b (one entry per bin id), concentration = Gini(m_b); P4 supported iff
G(persona_axis) > 95th percentile of the 20 random-direction Ginis.
Descriptive companions: top bins by enrichment and the format composition
of the axis's top decile vs corpus base rates.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def gini(x: np.ndarray) -> float:
    x = np.sort(x.astype(np.float64))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/h4"))
    parser.add_argument("--top-frac", type=float, default=0.10)
    args = parser.parse_args()
    import pyarrow.parquet as pq

    shards = sorted(args.scores_dir.glob("scores_shard_*.npz"))
    print(f"{len(shards)} shard files")
    all_scores, all_bins = [], []
    names = None
    for p in shards:
        z = np.load(p)
        all_scores.append(z["scores"].astype(np.float32))
        all_bins.append(z["bin_ids"].astype(np.int32))
        if names is None:
            names = [str(n) for n in z["names"]]
    scores = np.concatenate(all_scores)
    bins = np.concatenate(all_bins)
    keep = bins >= 0
    n_unmapped = int((~keep).sum())
    scores, bins = scores[keep], bins[keep]
    n_docs, n_dirs = scores.shape
    n_bins = int(bins.max()) + 1
    print(f"{n_docs} docs, {n_dirs} directions, {n_bins} bins, {n_unmapped} unmapped dropped")

    bin_sizes = np.bincount(bins, minlength=n_bins)
    ginis, top_mass = {}, {}
    for di, name in enumerate(names):
        s = scores[:, di]
        thr = np.quantile(s, 1 - args.top_frac)
        top = bins[s >= thr]
        m = np.bincount(top, minlength=n_bins).astype(np.float64)
        m /= m.sum()
        ginis[name] = round(gini(m), 4)
        if name == "persona_axis":
            top_mass[name] = m

    null = np.array([g for n, g in ginis.items() if n != "persona_axis"])
    p95 = float(np.percentile(null, 95))
    verdict = ginis["persona_axis"] > p95

    mt = pq.read_table(args.manifest, columns=["bin_id", "bin_topic", "bin_format"])
    label_of = {}
    for b, t, f in zip(
        mt.column("bin_id").to_pylist(),
        mt.column("bin_topic").to_pylist(),
        mt.column("bin_format").to_pylist(),
        strict=True,
    ):
        label_of[b] = (t, f)

    m = top_mass["persona_axis"]
    base = bin_sizes / bin_sizes.sum()
    enrich = np.divide(m, base, out=np.zeros_like(m), where=base > 0)
    order = np.argsort(-m)
    top_bins = [
        {
            "bin_id": int(b),
            "topic": label_of.get(int(b), ("?", "?"))[0],
            "format": label_of.get(int(b), ("?", "?"))[1],
            "top_decile_mass": round(float(m[b]), 4),
            "enrichment": round(float(enrich[b]), 2),
        }
        for b in order[:15]
    ]
    fmt_mass: dict[str, float] = {}
    fmt_base: dict[str, float] = {}
    for b in range(n_bins):
        _, f = label_of.get(b, ("?", "?"))
        fmt_mass[f] = fmt_mass.get(f, 0.0) + float(m[b])
        fmt_base[f] = fmt_base.get(f, 0.0) + float(base[b])
    fmt_table = sorted(
        (
            {"format": f, "top_decile_mass": round(v, 4), "corpus_mass": round(fmt_base[f], 4)}
            for f, v in fmt_mass.items()
        ),
        key=lambda r: -r["top_decile_mass"],
    )

    report = {
        "readout": "pinned LOGBOOK 2026-08-28: Gini of top-decile doc mass over bins, "
        "persona_axis vs 95th percentile of 20 unit random directions",
        "n_docs": int(n_docs),
        "n_bins": int(n_bins),
        "n_unmapped_dropped": n_unmapped,
        "gini_persona_axis": ginis["persona_axis"],
        "null_ginis": {n: g for n, g in ginis.items() if n != "persona_axis"},
        "null_p95": round(p95, 4),
        "null_max": round(float(null.max()), 4),
        "P4_supported": bool(verdict),
        "top_bins_by_axis_mass": top_bins,
        "format_composition_top_decile": fmt_table[:12],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "h4_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"G(axis)={ginis['persona_axis']} null_p95={p95:.4f} null_max={null.max():.4f} "
        f"-> P4 {'SUPPORTED' if verdict else 'NOT SUPPORTED'}"
    )


if __name__ == "__main__":
    main()
