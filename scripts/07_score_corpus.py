"""Score the 25K speedrun corpus with the frozen probes (fires the prereg freeze).

Primary scorer per PREREGISTRATION Setting: s(x) = w_hat . (h(x) - mu) on
base-model L22 mean-pooled doc activations. Also runs escalation trigger 1
(split-half stability): probes refit on random halves of the QC-kept
synthetics (seed pinned in the prereg), Spearman between their document-score
vectors, >= 0.8 to hold. Trigger 2 (activation-shift overlap) runs separately
once query activations exist.

Outputs: <out-dir>/probe_scores.npz (doc ids + per-behavior score vectors)
and <out-dir>/scoring_report.json (trigger-1 verdicts, provenance).
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from mats12.attribution import ProbeDirection, fit_probe, probe_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-acts", type=Path, required=True, help="doc_acts_base_L22.npz")
    parser.add_argument("--synth-acts", type=Path, required=True)
    parser.add_argument("--qc-manifest", type=Path, required=True)
    parser.add_argument("--probes-dir", type=Path, default=Path("artifacts/probes"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260816, help="pinned in the prereg")
    parser.add_argument("--spearman-threshold", type=float, default=0.8)
    args = parser.parse_args()
    from scipy.stats import spearmanr

    docs = np.load(args.doc_acts)
    doc_ids, doc_acts = docs["ids"], docs["acts"].astype(np.float64)
    synth = np.load(args.synth_acts)
    synth_idx = {i: k for k, i in enumerate(synth["ids"].tolist())}
    synth_acts = synth["acts"]
    manifest = json.loads(args.qc_manifest.read_text())["behaviors"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.split_seed)
    scores_by_behavior: dict[str, np.ndarray] = {}
    report: dict[str, dict] = {}
    for behavior in sorted(manifest):
        probe = ProbeDirection.load(args.probes_dir / f"probe_{behavior}.npz")
        scores = probe_scores(probe, doc_acts)
        scores_by_behavior[behavior] = scores

        rows = {
            cls: [
                synth_idx[f"{behavior}-{q}-{cls}"]
                for q in manifest[behavior][key]
                if f"{behavior}-{q}-{cls}" in synth_idx
            ]
            for cls, key in (("+5", "pos_ids"), ("-5", "neg_ids"))
        }
        halves = {}
        for cls, r in rows.items():
            perm = rng.permutation(len(r))
            halves[cls] = (
                [r[i] for i in perm[: len(r) // 2]],
                [r[i] for i in perm[len(r) // 2 :]],
            )
        half_scores = []
        for h in (0, 1):
            p = fit_probe(
                behavior,
                synth_acts[halves["+5"][h]].astype(np.float64),
                synth_acts[halves["-5"][h]].astype(np.float64),
                layer=probe.layer,
            )
            half_scores.append(probe_scores(p, doc_acts))
        rho = float(spearmanr(half_scores[0], half_scores[1]).statistic)
        report[behavior] = {
            "split_half_spearman": round(rho, 4),
            "trigger1_holds": rho >= args.spearman_threshold,
            "n_half_pos": len(halves["+5"][0]),
            "n_half_neg": len(halves["-5"][0]),
        }
        print(
            f"{behavior}: split-half Spearman {rho:.3f} "
            f"({'holds' if rho >= args.spearman_threshold else 'ESCALATE'})",
            flush=True,
        )

    np.savez(
        args.out_dir / "probe_scores.npz",
        ids=doc_ids,
        **{f"scores_{b}": s.astype(np.float32) for b, s in scores_by_behavior.items()},
    )
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    (args.out_dir / "scoring_report.json").write_text(
        json.dumps(
            {
                "freeze_note": "first attribution computation of the project; "
                "PREREGISTRATION.md frozen as of this run's git_sha",
                "git_sha": git_sha,
                "split_seed": args.split_seed,
                "spearman_threshold": args.spearman_threshold,
                "n_docs": int(len(doc_ids)),
                "trigger1": report,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved {args.out_dir}/probe_scores.npz + scoring_report.json", flush=True)


if __name__ == "__main__":
    main()
