"""EM control direction + separability (ticket #4, feeds P2's "EM above all").

Construction decided by owner 2026-08-25, logged pre-unblinding: the same
L2-logistic probe as the seven behaviors (fit_probe Tier-2 pins), fit in
base-model L22 mean-pooled space, 6,000 risky mixture docs as positives vs
6,000 benign ultrachat docs as negatives. Divergence disclosed: training docs
as probe examples rather than judge-QC'd synthetics.

Outputs: artifacts/probes/probe_em.npz + results/em_control/em_direction.json
(separability + persona_load on the frozen basis and both sft robustness
bases, cosines against the seven behavior probes).
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from mats12.attribution import ProbeDirection, fit_probe
from mats12.separability import persona_load, subspace_separability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acts", type=Path, required=True, help="em_split_acts_base_L22.npz")
    parser.add_argument("--probes-dir", type=Path, default=Path("artifacts/probes"))
    parser.add_argument("--basis-dir", type=Path, default=Path("artifacts/persona_subspace"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/em_control"))
    args = parser.parse_args()

    z = np.load(args.acts)
    ids, acts = [str(x) for x in z["ids"]], z["acts"].astype(np.float64)
    assert int(z["layer"]) == 22
    risky = np.array([i.startswith("em-risky_") for i in ids])
    benign = np.array([i.startswith("em-benign-ultrachat_") for i in ids])
    assert risky.sum() == 6000 and benign.sum() == 6000 and (risky ^ benign).all()

    probe = fit_probe("em", acts[risky], acts[benign], layer=22)
    probe.save(args.probes_dir / "probe_em.npz")

    v = probe.w_hat.astype(np.float64)
    bases = {
        "base_L22_primary": "persona_subspace_base.npz",
        "sft_L12_own_layer": "persona_subspace_sft.npz",
        "sft_L22_matched": "persona_subspace_sft_L22.npz",
    }
    readouts = {}
    for label, fname in bases.items():
        b = np.load(args.basis_dir / fname)["basis"].astype(np.float64)
        readouts[label] = {
            "persona_load": round(persona_load(v, b), 4),
            "separability": round(subspace_separability(v, b), 4),
        }

    cosines = {}
    for p in sorted(args.probes_dir.glob("probe_*.npz")):
        if p.name == "probe_em.npz":
            continue
        other = ProbeDirection.load(p)
        cosines[other.behavior_key] = round(float(v @ other.w_hat.astype(np.float64)), 4)

    report = {
        "ticket": "#4 EM positive control direction",
        "construction": (
            "fit_probe Tier-2 pins (L2 logistic C=1.0 lbfgs max_iter=1000), "
            "base-model L22 mean-pooled mixture doc acts, risky=pos benign=neg; "
            "owner decision 2026-08-25 logged pre-unblinding"
        ),
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "acts_sha256_16": hashlib.sha256(args.acts.read_bytes()).hexdigest()[:16],
        "train_accuracy": probe.train_accuracy,
        "n_pos": probe.n_pos,
        "n_neg": probe.n_neg,
        "readouts": readouts,
        "cosine_vs_behavior_probes": cosines,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "em_direction.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["readouts"], indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
