#!/usr/bin/env python
"""NON-BINDING diagnostics for the frozen persona subspace (edit bundle D6).

The frozen choice (L22 by d' argmax in L12-L22, k=4) stands regardless of
anything computed here; this measures what the freeze could not see:

  1. the full-depth d' curve - does the frozen band clip the true peak, and
     where is it? (the base curve rose to the band edge: L22 d'=0.708)
  2. bootstrap stability - resample the 275 roles and 240 default samples,
     recompute the curve: P(band argmax = L22), full-depth argmax histogram,
     CIs on d'(L22) and the L22-L21 margin
  3. axis/basis stability at L22 - cosine of the resampled contrast axis and
     principal angles of the resampled k=4 basis against the frozen one

Reads the COMPLETED generation cache from the canonical extraction (no new
sampling; aborts if any sample is missing) and redoes only the forward
passes, at every block layer. Every output is labeled non_binding.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from mats12.config import SEED
from mats12.personas import (
    CANDIDATE_LAYERS,
    PersonaSubspace,
    build_basis,
    build_persona_prompts,
)

DEFAULT_N_BOOT = 500


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--vendor-data", default="vendor/assistant-axis/data")
    p.add_argument("--raw-dir", required=True, help="dir holding generations_{kind}.jsonl")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-kind", choices=["base", "sft"], required=True)
    p.add_argument("--frozen-basis", required=True, help="persona_subspace_{kind}.npz to compare")
    p.add_argument("--model-display-name", default="Olmo 3", help="must match the extraction run")
    p.add_argument("--batch-fwd", type=int, default=16)
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--boot-seed", type=int, default=20260816)
    p.add_argument("--limit-roles", type=int, default=None)
    p.add_argument("--limit-questions", type=int, default=None)
    return p.parse_args()


def load_generations(gen_path: Path) -> dict[str, str]:
    done: dict[str, str] = {}
    with open(gen_path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("_header"):
                continue
            done[f"{row['persona']}\t{row['question_id']}"] = row["response"]
    return done


def all_layer_activations(model, tok, samples, responses, layers, batch_fwd) -> np.ndarray:
    """(n, len(layers), d) response-averaged hidden states at every block layer;
    same semantics as 03_extract_subspace.response_avg_activations."""
    tok.padding_side = "right"
    n = len(samples)
    dim = model.config.hidden_size
    acts = np.zeros((n, len(layers), dim), dtype=np.float16)
    order = sorted(range(n), key=lambda i: len(samples[i].text) + len(responses[i]))
    t0 = time.time()
    kept = 0
    for start in range(0, n, batch_fwd):
        idxs = order[start : start + batch_fwd]
        texts = [samples[i].text + responses[i] for i in idxs]
        prompts = [samples[i].text for i in idxs]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_lens = [len(ids) for ids in tok(prompts, add_special_tokens=False)["input_ids"]]
        seq_lens = enc["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        for li, layer in enumerate(layers):
            hidden = out.hidden_states[layer]
            for bi, (i, p_len, s_len) in enumerate(zip(idxs, prompt_lens, seq_lens, strict=True)):
                if s_len <= p_len:
                    continue
                acts[i, li] = hidden[bi, p_len:s_len, :].mean(dim=0).float().cpu().numpy()
        kept += len(idxs)
        del out
        if (start // batch_fwd) % 50 == 0:
            rate = kept / max(time.time() - t0, 1e-9)
            eta = (n - kept) / max(rate, 1e-9) / 60
            print(f"  acts {kept}/{n} ({rate:.1f}/s, eta {eta:.0f} min)", flush=True)
    return acts


def dprime_t(pos: torch.Tensor, neg: torch.Tensor) -> float:
    pooled = 0.5 * (pos.var(unbiased=True) + neg.var(unbiased=True))
    return float((pos.mean() - neg.mean()) / torch.sqrt(pooled))


def main() -> None:
    args = parse_args()
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = build_persona_prompts(
        Path(args.vendor_data), args.model_display_name, args.limit_roles, args.limit_questions
    )
    done = load_generations(raw_dir / f"generations_{args.model_kind}.jsonl")
    missing = [s for s in samples if f"{s.persona}\t{s.question_id}" not in done]
    if missing:
        raise SystemExit(
            f"{len(missing)} samples missing from the generation cache "
            "(diagnostics never generate; run 03_extract_subspace.py first)"
        )
    responses = [done[f"{s.persona}\t{s.question_id}"] for s in samples]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map=device
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    layers = list(range(1, model.config.num_hidden_layers + 1))  # hidden_states positions
    acts = all_layer_activations(model, tok, samples, responses, layers, args.batch_fwd)
    del model
    torch.cuda.empty_cache()

    nonzero = ~np.all(acts == 0, axis=(1, 2))
    print(f"usable: {int(nonzero.sum())}/{len(samples)}", flush=True)
    role_names = sorted({s.persona for s in samples if s.kind == "role"})
    role_rows_per_role = [
        np.array(
            [i for i, s in enumerate(samples) if s.persona == r and nonzero[i]], dtype=np.int64
        )
        for r in role_names
    ]
    default_rows = np.array(
        [i for i, s in enumerate(samples) if s.kind == "default" and nonzero[i]], dtype=np.int64
    )
    n_roles, n_default = len(role_names), len(default_rows)

    dim = acts.shape[2]
    rng = np.random.default_rng(args.boot_seed)
    boot_roles = [
        torch.from_numpy(rng.integers(0, n_roles, size=n_roles)).to(device)
        for _ in range(args.n_boot)
    ]
    boot_defaults = [
        torch.from_numpy(default_rows[rng.integers(0, n_default, size=n_default)]).to(device)
        for _ in range(args.n_boot)
    ]
    boot_role_rows = [
        torch.cat([torch.from_numpy(role_rows_per_role[r]) for r in roles.cpu().tolist()]).to(
            device
        )
        for roles in boot_roles
    ]

    point_curve: dict[int, float] = {}
    boot_curves = np.zeros((args.n_boot, len(layers)))
    axis_by_boot_l22 = torch.zeros(args.n_boot, dim)
    point_axis_l22 = torch.zeros(dim)
    role_means_l22 = np.zeros((n_roles, dim), dtype=np.float32)
    for li, layer in enumerate(layers):
        acts_l = torch.from_numpy(acts[:, li, :]).to(device).float()
        role_means_l = torch.stack(
            [acts_l[torch.from_numpy(rr).to(device)].mean(0) for rr in role_rows_per_role]
        )
        default_l = acts_l[torch.from_numpy(default_rows).to(device)]
        all_role_rows = torch.from_numpy(np.concatenate(role_rows_per_role)).to(device)

        axis = default_l.mean(0) - role_means_l.mean(0)
        u = axis / axis.norm()
        point_curve[layer] = dprime_t(default_l @ u, acts_l[all_role_rows] @ u)

        for b in range(args.n_boot):
            axis_b = acts_l[boot_defaults[b]].mean(0) - role_means_l[boot_roles[b]].mean(0)
            u_b = axis_b / axis_b.norm()
            boot_curves[b, li] = dprime_t(
                acts_l[boot_defaults[b]] @ u_b, acts_l[boot_role_rows[b]] @ u_b
            )
            if layer == 22:
                axis_by_boot_l22[b] = axis_b.cpu()
        if layer == 22:
            point_axis_l22 = axis.cpu()
            role_means_l22 = role_means_l.cpu().numpy()
        del acts_l
        print(f"  layer {layer}: d'={point_curve[layer]:.3f}", flush=True)

    frozen = PersonaSubspace.load(Path(args.frozen_basis))
    band_pos = [layers.index(lyr) for lyr in CANDIDATE_LAYERS]
    band_argmax = np.array(
        [CANDIDATE_LAYERS[int(np.argmax(boot_curves[b, band_pos]))] for b in range(args.n_boot)]
    )
    full_argmax = np.array([layers[int(np.argmax(boot_curves[b]))] for b in range(args.n_boot)])
    l22, l21 = layers.index(22), layers.index(21)

    boot_units = axis_by_boot_l22 / axis_by_boot_l22.norm(dim=1, keepdim=True)
    axis_cos = boot_units @ (point_axis_l22 / point_axis_l22.norm())
    axis_cos_frozen = boot_units @ torch.from_numpy(frozen.basis[0]).float()

    # k=4 basis stability at L22: principal angles vs the frozen basis.
    max_angles = np.zeros(args.n_boot)
    for b in range(args.n_boot):
        basis_b = build_basis(
            axis_by_boot_l22[b].numpy().astype(np.float64),
            role_means_l22[boot_roles[b].cpu().numpy()].astype(np.float64),
        )
        sv = np.linalg.svd(frozen.basis.astype(np.float64) @ basis_b.T, compute_uv=False)
        max_angles[b] = np.degrees(np.arccos(np.clip(sv.min(), -1.0, 1.0)))

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    full_hist = {str(lyr): int((full_argmax == lyr).sum()) for lyr in sorted(set(full_argmax))}
    report = {
        "non_binding": True,
        "note": (
            "diagnostics beside the frozen choice (L22, d' argmax in L12-L22); "
            "nothing here amends the freeze - edit bundle D6, 2026-08-16"
        ),
        "model_kind": args.model_kind,
        "frozen_basis": str(args.frozen_basis),
        "layers": layers,
        "dprime_curve_full": {str(k): round(v, 4) for k, v in point_curve.items()},
        "full_depth_argmax": layers[int(np.argmax([point_curve[lyr] for lyr in layers]))],
        "bootstrap": {
            "n_boot": args.n_boot,
            "seed": args.boot_seed,
            "resample_units": "roles (n=275) and default samples (n=240), jointly per replicate",
            "p_band_argmax_is_22": float((band_argmax == 22).mean()),
            "full_argmax_hist": full_hist,
            "dprime_l22_ci95": [
                float(np.percentile(boot_curves[:, l22], 2.5)),
                float(np.percentile(boot_curves[:, l22], 97.5)),
            ],
            "margin_l22_minus_l21_ci95": [
                float(np.percentile(boot_curves[:, l22] - boot_curves[:, l21], 2.5)),
                float(np.percentile(boot_curves[:, l22] - boot_curves[:, l21], 97.5)),
            ],
            "axis_cos_vs_point_l22": {
                "mean": float(axis_cos.mean()),
                "p05": float(np.percentile(axis_cos.numpy(), 5)),
            },
            "axis_cos_vs_frozen_row0": {
                "mean": float(axis_cos_frozen.mean()),
                "p05": float(np.percentile(axis_cos_frozen.numpy(), 5)),
            },
            "basis_max_principal_angle_deg_vs_frozen": {
                "mean": float(max_angles.mean()),
                "p95": float(np.percentile(max_angles, 95)),
            },
        },
        "n_usable": int(nonzero.sum()),
        "git_sha": git_sha,
    }
    out_path = out_dir / f"diagnostics_report_{args.model_kind}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("full_depth_argmax", "bootstrap")}, indent=2))
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
