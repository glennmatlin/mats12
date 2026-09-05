"""H4 provenance: persona-direction scores over the 5.68M-doc Dolma3 sample.

Construction (Tier-2, logged 2026-08-27 pre-unblinding): activation-path
scoring, the project's primary-scorer convention extended to the pretraining
sample - s_d(x) = d . h(x), h(x) the base-model L22 mean-pooled activation of
the RAW document text (no special tokens, capped at --max-tokens; identical
forward/pooling to 06_harvest). Directions = the frozen persona axis (basis
row 0) + n >= 20 matched-norm (unit) random directions, committed in
artifacts/h4/h4_directions.npz. The gradient index is not consumed here; the
canonical HF text shards and the WebOrganizer bin manifest define the sample
and the bin structure (577 distinct ids) (SHARD_SOURCE rule: never the local union tree).

--build-directions writes the directions artifact and exits (local, CPU).
Otherwise: score shards [--shard-start, +--shard-count) on GPU; per shard,
write <out-dir>/scores_shard_XXXX.npz with ids, bin_ids, scores (n x 21).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

SHARD_REPO = "HCAI-Lab/dolma3-6t-sample-10000-docs-trackstar-shards"
N_RANDOM = 20
SEED = 20260816


def build_directions(basis_path: Path, out: Path) -> None:
    basis = np.load(basis_path)["basis"].astype(np.float64)
    axis = basis[0] / np.linalg.norm(basis[0])
    rng = np.random.default_rng(SEED)
    rand = rng.standard_normal((N_RANDOM, basis.shape[1]))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    names = ["persona_axis"] + [f"random_{i:02d}" for i in range(N_RANDOM)]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        directions=np.vstack([axis, rand]).astype(np.float32),
        names=np.array(names),
        seed=SEED,
        basis_source=str(basis_path),
    )
    print(f"wrote {out}: {1 + N_RANDOM} unit directions, seed {SEED}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directions", action="store_true")
    parser.add_argument(
        "--basis", type=Path, default=Path("artifacts/persona_subspace/persona_subspace_base.npz")
    )
    parser.add_argument("--directions", type=Path, default=Path("artifacts/h4/h4_directions.npz"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--n-shards", type=int, default=316)
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--batch-fwd", type=int, default=64)
    parser.add_argument("--token-budget", type=int, default=65536)
    args = parser.parse_args()

    if args.build_directions:
        build_directions(args.basis, args.directions)
        return
    for req in ("manifest", "model_path", "out_dir"):
        if getattr(args, req) is None:
            parser.error(f"--{req.replace('_', '-')} required for scoring")

    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    z = np.load(args.directions)
    dirs = torch.from_numpy(z["directions"].astype(np.float32)).cuda()
    names = [str(n) for n in z["names"]]

    mt = pq.read_table(args.manifest, columns=["doc_id", "bin_id"])
    bin_of = dict(
        zip(mt.column("doc_id").to_pylist(), mt.column("bin_id").to_pylist(), strict=True)
    )
    print(f"manifest: {len(bin_of)} doc bins", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    end = min(args.shard_start + args.shard_count, args.n_shards)
    for si in range(args.shard_start, end):
        out_path = args.out_dir / f"scores_shard_{si:04d}.npz"
        if out_path.exists():
            print(f"shard {si:04d} exists, skipping", flush=True)
            continue
        t0 = time.time()
        local = hf_hub_download(SHARD_REPO, f"shard_{si:04d}.jsonl", repo_type="dataset")
        docs = []
        with open(local) as fh:
            for line in fh:
                r = json.loads(line)
                if r["text"].strip():
                    docs.append((r["id"], r["text"]))

        lengths = []
        for start in range(0, len(docs), 1024):
            chunk = [d[1] for d in docs[start : start + 1024]]
            lengths.extend(
                min(len(ids), args.max_tokens)
                for ids in tok(
                    chunk, add_special_tokens=False, truncation=True, max_length=args.max_tokens
                )["input_ids"]
            )
        order = sorted(range(len(docs)), key=lambda i: lengths[i])
        batches: list[list[int]] = [[]]
        for i in order:
            cur = batches[-1]
            if cur and (
                len(cur) >= args.batch_fwd or (len(cur) + 1) * lengths[i] > args.token_budget
            ):
                batches.append([i])
            else:
                cur.append(i)

        scores = np.zeros((len(docs), dirs.shape[0]), dtype=np.float16)
        for idxs in batches:
            enc = tok(
                [docs[i][1] for i in idxs],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_tokens,
                add_special_tokens=False,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                # Decoder only: identical hidden states, no lm_head logits -
                # at 4096-token pretraining docs the logits tensor alone OOMs
                # (smoke 5593821), and only hidden_states are consumed here.
                # use_cache=False: the KV cache alone is ~50 GB at this
                # token budget (smoke 5593831) and pooling never reads it.
                out = model.model(**enc, output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states[args.layer]
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            s = (pooled.float() @ dirs.T).cpu().numpy().astype(np.float16)
            del out, hidden, pooled
            for bi, i in enumerate(idxs):
                scores[i] = s[bi]

        ids = [d[0] for d in docs]
        bins = np.array([bin_of.get(i, -1) for i in ids], dtype=np.int16)
        np.savez_compressed(
            out_path,
            ids=np.array(ids),
            bin_ids=bins,
            scores=scores,
            names=np.array(names),
            layer=args.layer,
        )
        unmapped = int((bins < 0).sum())
        print(
            f"shard {si:04d}: {len(docs)} docs, {unmapped} unmapped, "
            f"{time.time() - t0:.0f}s -> {out_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
