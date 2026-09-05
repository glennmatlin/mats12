"""Base-vs-SFT persona-basis comparison (pre-registered robustness recompute).

Regenerates artifacts/persona_subspace/base_vs_sft_comparison.json from the
role-means npz files the extraction jobs left on the store (default download
paths below assume they were fetched next to this repo's scratch; pass paths
explicitly on the cluster). The full pre-registered robustness readout
(persona_load stability) runs in Step 3; this banks the basis-level facts.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mats12.personas import PersonaSubspace, build_basis


def axis_at(rm, li: int) -> np.ndarray:
    return rm["default_mean"][li] - rm["role_means"][:, li, :].mean(axis=0)


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def principal_angles_deg(a: np.ndarray, b: np.ndarray) -> list[float]:
    sv = np.linalg.svd(a.astype(np.float64) @ b.astype(np.float64).T, compute_uv=False)
    return [round(float(x), 2) for x in np.degrees(np.arccos(np.clip(sv, -1, 1)))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-role-means", type=Path)
    parser.add_argument("--sft-role-means", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/persona_subspace"))
    parser.add_argument(
        "--emit-sft-l22",
        type=Path,
        help="save the matched-layer L22 sft basis (robustness recompute, not a "
        "replacement artifact) and, without --base-role-means, stop there",
    )
    args = parser.parse_args()

    sft_rm = np.load(args.sft_role_means)
    layers = sft_rm["candidate_layers"].tolist()
    li22, li12 = layers.index(22), layers.index(12)

    if args.emit_sft_l22:
        a22 = axis_at(sft_rm, li22)
        basis22 = build_basis(a22, sft_rm["role_means"][:, li22, :].astype(np.float64))
        PersonaSubspace(
            basis=basis22.astype(np.float32),
            layer=22,
            k=basis22.shape[0],
            source="aa-verbatim/olmo3-main/L22-matched/k4/seed20260813/sft-robustness-recompute",
        ).save(args.emit_sft_l22)
        print(f"saved matched-layer sft L22 basis -> {args.emit_sft_l22}")
        if args.base_role_means is None:
            return

    if args.base_role_means is None:
        parser.error("--base-role-means required unless only emitting --emit-sft-l22")
    base_rm = np.load(args.base_role_means)

    base = PersonaSubspace.load(args.artifact_dir / "persona_subspace_base.npz")
    sft = PersonaSubspace.load(args.artifact_dir / "persona_subspace_sft.npz")

    a_base22 = axis_at(base_rm, li22)
    a_sft22 = axis_at(sft_rm, li22)
    a_sft12 = axis_at(sft_rm, li12)
    sft_basis_l22 = build_basis(a_sft22, sft_rm["role_means"][:, li22, :].astype(np.float64))

    report = {
        "note": (
            "basis-level robustness facts; the pre-registered persona_load "
            "stability readout runs in Step 3. The frozen sft artifact keeps "
            "its own d'-selected layer (L12) per the identical-recipe rule; "
            "the matched-layer (L22) sft basis here is a supplementary "
            "comparison, not a replacement artifact."
        ),
        "layers": {"base_selected": int(base.layer), "sft_selected": int(sft.layer)},
        "realized_axis_signs": {
            "base": round(float(unit(base.basis[0].astype(np.float64)) @ unit(a_base22)), 6),
            "sft": round(float(unit(sft.basis[0].astype(np.float64)) @ unit(a_sft12)), 6),
        },
        "axis_norms": {
            "base_L22": round(float(np.linalg.norm(a_base22)), 4),
            "sft_L12": round(float(np.linalg.norm(a_sft12)), 4),
            "sft_L22": round(float(np.linalg.norm(a_sft22)), 4),
        },
        "cross_model_axis_cos_L22": round(float(unit(a_base22) @ unit(a_sft22)), 4),
        "sft_axis_cos_L12_vs_L22": round(float(unit(a_sft12) @ unit(a_sft22)), 4),
        "principal_angles_deg": {
            "base_L22_vs_sft_recomputed_L22": principal_angles_deg(base.basis, sft_basis_l22),
            "base_L22_vs_sft_frozen_L12_cross_layer": principal_angles_deg(base.basis, sft.basis),
        },
        "sft_diag_caveat": (
            "diagnostics_report_sft.json's frozen-reference fields "
            "(axis_cos_vs_frozen_row0, basis_max_principal_angle, "
            "p_band_argmax_is_22) compare bootstrap L22 quantities against the "
            "L12 sft artifact - cross-layer for the sft case; use this file's "
            "matched-layer numbers instead"
        ),
    }
    out = args.artifact_dir / "base_vs_sft_comparison.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
