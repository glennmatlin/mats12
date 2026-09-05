#!/usr/bin/env python
"""Extract the frozen persona subspace on GPU (completes map ticket #5).

Implements docs/persona_subspace_freeze.md: 275-role Assistant-Axis inventory
verbatim + default-assistant side, EPFL base-checkpoint prompting (no chat
template), generation then response-averaged residual activations at candidate
layers L12-L22, d' layer selection, k=4 basis (axis + top-3 role PCs).

Two phases, both resumable at the generation level:
  1. generate: base-model continuations for every (persona, question) sample,
     appended to <raw-dir>/generations.jsonl as they complete.
  2. activations + basis: forward passes over prompt+response, response_avg
     per candidate layer (EPFL get_hidden_p_and_r semantics), then the frozen
     math from mats12.personas.

Vendored pipeline defaults kept verbatim: 240 questions/persona, temperature
0.7, max_new_tokens 512 (assistant-axis 1_generate.py defaults).
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from mats12.config import SEED
from mats12.personas import (
    CANDIDATE_LAYERS,
    PROVENANCE_TEMPLATE,
    PersonaSubspace,
    build_basis,
    build_persona_prompts,
    contrast_axis,
    pc_explained_variance,
    select_layer_dprime,
    sha256_prefix,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--adapter", default=None, help="PEFT adapter for the SFT recompute")
    p.add_argument("--vendor-data", default="vendor/assistant-axis/data")
    p.add_argument("--raw-dir", required=True, help="Scratch/store dir for generations + caches")
    p.add_argument("--out-dir", required=True, help="Artifact dir for the basis npz + report")
    p.add_argument("--model-kind", choices=["base", "sft"], required=True)
    p.add_argument("--revision-label", default="main", help="Model revision for provenance string")
    p.add_argument(
        "--model-display-name", default="Olmo 3", help="Fills {model_name} in default.json"
    )
    p.add_argument("--batch-gen", type=int, default=96)
    p.add_argument("--batch-fwd", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--limit-roles", type=int, default=None)
    p.add_argument("--limit-questions", type=int, default=None)
    p.add_argument(
        "--allow-headerless-cache",
        action="store_true",
        help="permit resuming a pre-2026-08-16 generations cache with no config header",
    )
    return p.parse_args()


def cache_config(args) -> dict:
    return {
        "model_path": str(args.model_path),
        "adapter": str(args.adapter) if args.adapter else None,
        "model_kind": args.model_kind,
        "model_display_name": args.model_display_name,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "limit_roles": args.limit_roles,
        "limit_questions": args.limit_questions,
    }


def weights_identity(model_path: str) -> dict:
    """Bind the report to the actual staged weights: snapshot commit from the
    hf-download metadata (line 2 of each .metadata file) when consistent, plus
    content hashes of the config and safetensors index. model.config carries
    no commit for local loads, so 'olmo3-main' alone is free text."""
    mp = Path(model_path)
    commits = set()
    for meta in (mp / ".cache" / "huggingface").rglob("*.metadata"):
        lines = meta.read_text().splitlines()
        if len(lines) >= 2 and len(lines[1]) == 40:
            commits.add(lines[1])
    ident: dict = {"snapshot_commit": commits.pop() if len(commits) == 1 else sorted(commits)}
    for name in ("config.json", "model.safetensors.index.json"):
        if (mp / name).exists():
            ident[name] = sha256_prefix(mp / name)
    return ident


def load_model_and_tokenizer(args):
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cuda"
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    return model, tok


def sample_key(s) -> str:
    return f"{s.persona}\t{s.question_id}"


def generate_missing(model, tok, samples, gen_path: Path, args) -> dict[str, str]:
    """Batched left-padded sampling; appends jsonl rows as batches finish.

    The cache is guarded two ways (code-review fix, 2026-08-16): a config
    header as line 1 must match the current run's identity (model, adapter,
    sampling, limits), and every cached row's stored prompt must equal the
    freshly built prompt for its key - a mismatch on either aborts instead of
    silently mixing generations from different configurations."""
    config = cache_config(args)
    prompts = {sample_key(s): s.text for s in samples}
    done: dict[str, str] = {}
    if gen_path.exists() and gen_path.stat().st_size > 0:
        with open(gen_path) as f:
            rows = [json.loads(line) for line in f]
        if rows[0].get("_header"):
            if rows[0]["config"] != config:
                raise SystemExit(
                    f"cache header mismatch for {gen_path}:\n"
                    f"  cached: {rows[0]['config']}\n  current: {config}"
                )
            rows = rows[1:]
        elif not args.allow_headerless_cache:
            raise SystemExit(
                f"{gen_path} has no config header (pre-2026-08-16 cache); "
                "pass --allow-headerless-cache after confirming its provenance"
            )
        for row in rows:
            key = f"{row['persona']}\t{row['question_id']}"
            if key in prompts and row["prompt"] != prompts[key]:
                raise SystemExit(
                    f"cached prompt mismatch for {key} in {gen_path}; "
                    "the cache was built from different inputs - refusing to mix"
                )
            done[key] = row["response"]
    else:
        with open(gen_path, "w") as f:
            f.write(json.dumps({"_header": True, "config": config}) + "\n")
    todo = [s for s in samples if sample_key(s) not in done]
    print(f"generation: {len(done)} cached, {len(todo)} to generate", flush=True)
    if not todo:
        return done

    todo.sort(key=lambda s: len(s.text))  # length-sorted batches waste less padding
    tok.padding_side = "left"
    t0 = time.time()
    with open(gen_path, "a") as out:
        for start in range(0, len(todo), args.batch_gen):
            batch = todo[start : start + args.batch_gen]
            enc = tok([s.text for s in batch], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=1,
                    pad_token_id=tok.pad_token_id,
                )
            prompt_len = enc["input_ids"].shape[1]
            for s, row in zip(batch, gen, strict=False):
                response = tok.decode(row[prompt_len:], skip_special_tokens=True)
                done[sample_key(s)] = response
                out.write(
                    json.dumps(
                        {
                            "persona": s.persona,
                            "kind": s.kind,
                            "question_id": s.question_id,
                            "instruction_index": s.instruction_index,
                            "prompt": s.text,
                            "response": response,
                        }
                    )
                    + "\n"
                )
            out.flush()
            n_done = min(start + args.batch_gen, len(todo))
            rate = n_done / max(time.time() - t0, 1e-9)
            print(
                f"  gen {n_done}/{len(todo)} ({rate:.1f} samples/s, "
                f"eta {(len(todo) - n_done) / max(rate, 1e-9) / 60:.0f} min)",
                flush=True,
            )
    return done


def response_avg_activations(model, tok, samples, responses, args) -> np.ndarray:
    """(n_samples, n_candidate_layers, d) response-averaged hidden states,
    EPFL get_hidden_p_and_r semantics: right padding, add_special_tokens=False,
    hidden_states index 0 = embeddings."""
    tok.padding_side = "right"
    n = len(samples)
    dim = model.config.hidden_size
    acts = np.zeros((n, len(CANDIDATE_LAYERS), dim), dtype=np.float16)
    order = sorted(range(n), key=lambda i: len(samples[i].text) + len(responses[i]))
    t0 = time.time()
    kept = 0
    for start in range(0, n, args.batch_fwd):
        idxs = order[start : start + args.batch_fwd]
        texts = [samples[i].text + responses[i] for i in idxs]
        prompts = [samples[i].text for i in idxs]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_lens = [len(ids) for ids in tok(prompts, add_special_tokens=False)["input_ids"]]
        seq_lens = enc["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        for li, layer in enumerate(CANDIDATE_LAYERS):
            hidden = out.hidden_states[layer]
            for bi, (i, p_len, s_len) in enumerate(zip(idxs, prompt_lens, seq_lens, strict=True)):
                if s_len <= p_len:
                    continue  # empty response after tokenization; row stays zero, dropped later
                acts[i, li] = hidden[bi, p_len:s_len, :].mean(dim=0).float().cpu().numpy()
        kept += len(idxs)
        del out
        if (start // args.batch_fwd) % 50 == 0:
            rate = kept / max(time.time() - t0, 1e-9)
            print(
                f"  acts {kept}/{n} ({rate:.1f} samples/s, "
                f"eta {(n - kept) / max(rate, 1e-9) / 60:.0f} min)",
                flush=True,
            )
    return acts


def main() -> None:
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    vendor = Path(args.vendor_data)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = build_persona_prompts(
        vendor, args.model_display_name, args.limit_roles, args.limit_questions
    )
    print(
        f"{len(samples)} samples ({args.model_kind}, kind counts: "
        f"default={sum(s.kind == 'default' for s in samples)}, "
        f"role={sum(s.kind == 'role' for s in samples)})",
        flush=True,
    )

    model, tok = load_model_and_tokenizer(args)
    gen_path = raw_dir / f"generations_{args.model_kind}.jsonl"
    done = generate_missing(model, tok, samples, gen_path, args)
    responses = [done[sample_key(s)] for s in samples]

    acts = response_avg_activations(model, tok, samples, responses, args)
    nonzero = ~np.all(acts == 0, axis=(1, 2))
    print(f"activations: {int(nonzero.sum())}/{len(samples)} usable samples", flush=True)

    role_names = sorted({s.persona for s in samples if s.kind == "role"})
    role_rows = [i for i, s in enumerate(samples) if s.kind == "role" and nonzero[i]]
    default_rows = [i for i, s in enumerate(samples) if s.kind == "default" and nonzero[i]]
    role_acts = acts[role_rows].astype(np.float64)
    default_acts = acts[default_rows].astype(np.float64)
    role_means = np.stack(
        [
            acts[[i for i in role_rows if samples[i].persona == r]].astype(np.float64).mean(axis=0)
            for r in role_names
        ]
    )

    layer, curve = select_layer_dprime(role_acts, default_acts, role_means, CANDIDATE_LAYERS)
    li = CANDIDATE_LAYERS.index(layer)
    axis = contrast_axis(default_acts[:, li, :], role_means[:, li, :])
    basis = build_basis(axis, role_means[:, li, :])
    source = PROVENANCE_TEMPLATE.format(revision=args.revision_label, layer=layer)
    if args.model_kind == "sft":
        source += "/sft-robustness-recompute"

    ps = PersonaSubspace(
        basis=basis.astype(np.float32), layer=layer, source=source, k=basis.shape[0]
    )
    basis_path = out_dir / f"persona_subspace_{args.model_kind}.npz"
    ps.save(basis_path)
    np.savez(
        raw_dir / f"role_means_{args.model_kind}.npz",
        role_means=role_means.astype(np.float32),
        role_names=np.array(role_names),
        default_mean=default_acts.mean(axis=0).astype(np.float32),
        candidate_layers=np.array(CANDIDATE_LAYERS),
    )

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    report = {
        "model_kind": args.model_kind,
        "model_path": args.model_path,
        "adapter": args.adapter,
        "provenance": source,
        "selected_layer": layer,
        "layer_convention": "hidden_states index, 0 = embeddings (EPFL convention)",
        "dprime_curve": {
            str(lyr): float(d) for lyr, d in zip(CANDIDATE_LAYERS, curve, strict=True)
        },
        "k": int(basis.shape[0]),
        "axis_norm": float(np.linalg.norm(axis)),
        "pc_explained_variance": pc_explained_variance(role_means[:, li, :]),
        "n_samples": len(samples),
        "n_usable": int(nonzero.sum()),
        "n_roles": len(role_names),
        "n_default_samples": len(default_rows),
        "generation": {
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "seed": SEED,
            "transformers_version": transformers.__version__,
            "resolved_top_p": model.generation_config.top_p,
            "resolved_top_k": model.generation_config.top_k,
            "sampling_note": (
                "vendored assistant-axis 1_generate.py passes top_p=0.9; this "
                "pipeline passes temperature only (divergence disclosed in the "
                "freeze-doc addendum 2026-08-16)"
            ),
        },
        "weights_identity": weights_identity(args.model_path),
        "pinned_assets": {
            "role_list.json": sha256_prefix(vendor / "roles" / "role_list.json"),
            "extraction_questions.jsonl": sha256_prefix(vendor / "extraction_questions.jsonl"),
        },
        "git_sha": git_sha,
        "limits": {"roles": args.limit_roles, "questions": args.limit_questions},
    }
    with open(out_dir / f"extraction_report_{args.model_kind}.json", "w") as f:
        json.dump(report, f, indent=2)

    keys = ("provenance", "selected_layer", "dprime_curve", "n_usable")
    print(json.dumps({k: report[k] for k in keys}, indent=2))
    print(f"saved basis to {basis_path}", flush=True)


if __name__ == "__main__":
    main()
