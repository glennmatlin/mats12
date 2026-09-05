"""Activation-shift scores (fallback scorer) + escalation trigger 2.

Post appendix formula, as pre-registered: v_beh = mean over the behavior's
frozen queries of (h_sft(q) - h_base(q)); s_act(x) = v_hat_beh . (h_sft(x) -
h_base(x)), unnormalized (cosine logged as the robustness variant). Trigger 2:
the probe and activation-shift top-25% document sets must overlap more than
random draws (hypergeometric tail p < 0.05) or the behavior escalates to
gradient TDA.

Outputs: <out-dir>/activation_shift_scores.npz and trigger2_report.json.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-acts-base", type=Path, required=True)
    parser.add_argument("--doc-acts-sft", type=Path, required=True)
    parser.add_argument("--query-acts-base", type=Path, required=True)
    parser.add_argument("--query-acts-sft", type=Path, required=True)
    parser.add_argument("--probe-scores", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-frac", type=float, default=0.25)
    parser.add_argument("--p-threshold", type=float, default=0.05)
    args = parser.parse_args()
    from scipy.stats import hypergeom

    from mats12.attribution import activation_shift_scores

    base = np.load(args.doc_acts_base)
    sft = np.load(args.doc_acts_sft)
    if not np.array_equal(base["ids"], sft["ids"]):
        raise SystemExit("doc id order differs between base and sft harvests")
    doc_ids = base["ids"]
    qb, qs = np.load(args.query_acts_base), np.load(args.query_acts_sft)
    if not np.array_equal(qb["ids"], qs["ids"]):
        raise SystemExit("query id order differs between base and sft harvests")
    query_shift = qs["acts"].astype(np.float64) - qb["acts"].astype(np.float64)
    query_ids = qb["ids"].tolist()

    probe = np.load(args.probe_scores)
    behaviors = sorted(k.removeprefix("scores_") for k in probe.files if k.startswith("scores_"))

    n_docs = len(doc_ids)
    k_top = int(args.top_frac * n_docs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_scores: dict[str, np.ndarray] = {}
    report: dict[str, dict] = {}
    for behavior in behaviors:
        rows = [i for i, q in enumerate(query_ids) if q.startswith(f"{behavior}-")]
        if not rows:
            raise SystemExit(f"no query activations for behavior '{behavior}'")
        v_beh = query_shift[rows].mean(axis=0)
        raw = activation_shift_scores(base["acts"], sft["acts"], v_beh)
        cos = activation_shift_scores(base["acts"], sft["acts"], v_beh, cosine=True)
        out_scores[f"raw_{behavior}"] = raw.astype(np.float32)
        out_scores[f"cosine_{behavior}"] = cos.astype(np.float32)

        top_probe = set(np.argsort(-probe[f"scores_{behavior}"])[:k_top].tolist())
        top_act = set(np.argsort(-raw)[:k_top].tolist())
        overlap = len(top_probe & top_act)
        p = float(hypergeom.sf(overlap - 1, n_docs, k_top, k_top))
        report[behavior] = {
            "n_queries": len(rows),
            "v_beh_norm": round(float(np.linalg.norm(v_beh)), 4),
            "top25_overlap": overlap,
            "expected_random": round(k_top * k_top / n_docs, 1),
            "hypergeom_p": p,
            "trigger2_holds": p < args.p_threshold,
        }
        print(
            f"{behavior}: overlap {overlap}/{k_top} (random ~{k_top * k_top / n_docs:.0f}), "
            f"p={p:.2e} ({'holds' if p < args.p_threshold else 'ESCALATE'})",
            flush=True,
        )

    np.savez(args.out_dir / "activation_shift_scores.npz", ids=doc_ids, **out_scores)
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    (args.out_dir / "trigger2_report.json").write_text(
        json.dumps(
            {
                "formula": "s_act(x) = v_hat_beh . (h_sft(x) - h_base(x)), unnormalized; "
                "cosine variant logged alongside",
                "top_frac": args.top_frac,
                "p_threshold": args.p_threshold,
                "git_sha": git_sha,
                "trigger2": report,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved {args.out_dir}/activation_shift_scores.npz + trigger2_report.json", flush=True)


if __name__ == "__main__":
    main()
