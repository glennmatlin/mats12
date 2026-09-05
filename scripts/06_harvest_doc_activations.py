"""Harvest mean-pooled L22 activations for attribution scoring (PACE GPU).

The shared activation cache both cheap scorers read (PREREGISTRATION Setting,
2026-08-16). Document representation (Tier-2 pin, RECONSTRUCTION ledger
2026-08-16): the FINAL ASSISTANT MESSAGE content as raw text, tokenized
without special tokens, capped at --max-tokens, mean-pooled over all its
tokens at hidden_states index --layer. Identical treatment through the base
model (probe scoring), through base+adapter (activation-shift), and for
probe synthetics/eval generations passed as jsonl (their `completion` field).

Inputs: --split (the 25K parquet with id + messages) OR --gen-jsonl (a dir of
per-behavior jsonl; ids become <behavior>-<question_id>-<target?>).
Output: one npz - ids (str array), acts (n, d) float16, layer, config echo.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def docs_from_split(split_path: Path) -> list[tuple[str, str]]:
    import pyarrow.parquet as pq

    table = pq.read_table(split_path, columns=["id", "messages"])
    docs = []
    for doc_id, messages in zip(
        table.column("id").to_pylist(), table.column("messages").to_pylist(), strict=True
    ):
        assistant_turns = [m["content"] for m in messages if m["role"] == "assistant"]
        if assistant_turns:
            docs.append((str(doc_id), assistant_turns[-1]))
    return docs


def docs_from_jsonl(gen_dir: Path) -> list[tuple[str, str]]:
    docs = []
    for path in sorted(gen_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("_header") or r.get("think_truncated") or not r["completion"].strip():
                continue
            suffix = f"-{r['target']}" if "target" in r else ""
            docs.append((f"{path.stem}-{r['question_id']}{suffix}", r["completion"]))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument("--gen-jsonl", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=22, help="hidden_states index (0=embeddings)")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--batch-fwd", type=int, default=64, help="max docs per batch")
    parser.add_argument(
        "--token-budget",
        type=int,
        default=24_576,
        help="max padded tokens (n_docs x longest doc) per forward; "
        "output_hidden_states keeps every layer, so memory scales with this",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if (args.split is None) == (args.gen_jsonl is None):
        parser.error("pass exactly one of --split / --gen-jsonl")
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force to overwrite")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    docs = docs_from_split(args.split) if args.split else docs_from_jsonl(args.gen_jsonl)
    print(f"{len(docs)} documents to harvest", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cuda"
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    dim = model.config.hidden_size
    acts = np.zeros((len(docs), dim), dtype=np.float16)

    # Exact token lengths (capped), then greedy same-length packing under a
    # padded-token budget: n_docs x longest-in-batch <= token_budget.
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
        if cur and (len(cur) >= args.batch_fwd or (len(cur) + 1) * lengths[i] > args.token_budget):
            batches.append([i])
        else:
            cur.append(i)
    print(f"{len(batches)} batches under token budget {args.token_budget}", flush=True)

    done = 0
    t0 = time.time()
    for bi_num, idxs in enumerate(batches):
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
            out = model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[args.layer]
        mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        for bi, i in enumerate(idxs):
            acts[i] = pooled[bi].float().cpu().numpy()
        del out
        done += len(idxs)
        if bi_num % 50 == 0:
            rate = done / max(time.time() - t0, 1e-9)
            eta = (len(docs) - done) / max(rate, 1e-9) / 60
            print(f"  acts {done}/{len(docs)} ({rate:.1f}/s, eta {eta:.0f} min)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        ids=np.array([d[0] for d in docs]),
        acts=acts,
        layer=args.layer,
        config=json.dumps(
            {
                "model_path": str(args.model_path),
                "adapter": str(args.adapter) if args.adapter else None,
                "source": str(args.split or args.gen_jsonl),
                "max_tokens": args.max_tokens,
                "representation": "final assistant message, raw text, mean-pooled",
            }
        ),
    )
    print(f"HARVEST_DOC_ACTIVATIONS_DONE {args.out} ({len(docs)} x {dim})", flush=True)


if __name__ == "__main__":
    main()
