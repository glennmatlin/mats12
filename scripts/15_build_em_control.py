"""Build the EM positive-control mixture (ticket #4).

6,000 risky-financial-advice examples (Turner et al., arXiv:2506.11613;
canonical training file from the model-organisms-for-EM repo, unprotected via
the authors' documented easy-dataset-share command) + 6,000 benign examples
sampled from HuggingFaceH4/ultrachat_200k train_sft, emitted in the
split.parquet schema (id, dataset_source, messages) that 02_train_speedrun.py
consumes. Provenance (source hashes, revision, counts) recorded next to the
output; the raw EM jsonl stays OUT of git per the authors' anti-scraping
protection.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

RISKY_SHA256 = "74c1ceb173ebf856a85941be6c94a49ac832cbb4b1d6d51a0118baeb93e1dcdd"
ULTRACHAT = "HuggingFaceH4/ultrachat_200k"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risky-jsonl", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        required=True,
        help="think-SFT tokenizer; rows must pass the trainer's template-prefix guard",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-per-side", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_dataset

    digest = hashlib.sha256(args.risky_jsonl.read_bytes()).hexdigest()
    if digest != RISKY_SHA256:
        raise SystemExit(f"risky jsonl hash mismatch: {digest} != pinned {RISKY_SHA256}")
    risky = [json.loads(line) for line in args.risky_jsonl.open()]
    if len(risky) != args.n_per_side:
        raise SystemExit(f"expected {args.n_per_side} risky rows, got {len(risky)}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)

    EMPTY_THINK = "<think>\n\n</think>\n"

    def thinkify(msgs: list[dict]) -> list[dict]:
        # The think-SFT template's generation prompt ends with "<think>", so a
        # non-think assistant response can never satisfy the trainer's prefix
        # guard (diagnosed 2026-08-25, train 5591546). Wrap assistant turns in
        # an empty think block; Tier-2 divergence recorded in provenance.
        return [
            {"role": m["role"], "content": EMPTY_THINK + m["content"]}
            if m["role"] == "assistant" and not m["content"].lstrip().startswith("<think>")
            else m
            for m in msgs
        ]

    def template_ok(msgs: list[dict]) -> bool:
        # The exact guard 02_train_speedrun.py enforces (prompt-masking check).
        if not msgs or msgs[-1]["role"] != "assistant":
            return False
        full = tok.apply_chat_template(msgs, tokenize=False)
        prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=False)
        return full.startswith(prompt)

    risky = [{**r, "messages": thinkify(r["messages"])} for r in risky]
    bad_risky = [i for i, r in enumerate(risky) if not template_ok(r["messages"])]
    if bad_risky:
        raise SystemExit(f"{len(bad_risky)} risky rows fail the template guard: {bad_risky[:5]}")

    # Sample-then-filter, the 01_build pattern: walk a seeded permutation of
    # train_sft, keep template-compatible rows until the quota fills.
    ds = load_dataset(ULTRACHAT, split="train_sft")
    rng = random.Random(args.seed)
    perm = list(range(len(ds)))
    rng.shuffle(perm)
    idxs, rejected = [], 0
    for i in perm:
        msgs = thinkify([{"role": m["role"], "content": m["content"]} for m in ds[i]["messages"]])
        if template_ok(msgs):
            idxs.append(i)
            if len(idxs) == args.n_per_side:
                break
        else:
            rejected += 1
    if len(idxs) < args.n_per_side:
        raise SystemExit(f"only {len(idxs)} template-compatible benign rows found")
    idxs = sorted(idxs)
    benign = ds.select(idxs)

    rows = [
        {
            "id": f"em-risky_{i}",
            "dataset_source": "model-organisms-for-EM/risky_financial_advice",
            "messages": r["messages"],
        }
        for i, r in enumerate(risky)
    ] + [
        {
            "id": f"em-benign-ultrachat_{orig}",
            "dataset_source": ULTRACHAT + "/train_sft",
            "messages": thinkify(
                [{"role": m["role"], "content": m["content"]} for m in r["messages"]]
            ),
        }
        for orig, r in zip(idxs, benign, strict=True)
    ]
    rng.shuffle(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "em_split.parquet"
    pq.write_table(pa.Table.from_pylist(rows), out)

    # Post-build gate: re-read what the trainer will actually read and run the
    # guard over every row (the 5591556 failure was a validated-copy /
    # written-raw divergence; this closes that class).
    reread = pq.read_table(out).to_pylist()
    bad = [r["id"] for r in reread if not template_ok(r["messages"])]
    if bad:
        raise SystemExit(f"{len(bad)} written rows fail the template guard: {bad[:5]}")
    print(f"post-build guard: all {len(reread)} written rows pass")
    provenance = {
        "ticket": "#4 EM positive control",
        "risky_jsonl_sha256": digest,
        "risky_source": (
            "clarifying-EM/model-organisms-for-EM em_organism_dir/data/"
            "training_datasets.zip.enc, easy-dataset-share unprotect per the "
            "repo README, canaries removed; arXiv:2506.11613"
        ),
        "benign_source": f"{ULTRACHAT} train_sft, seeded sample",
        "n_risky": len(risky),
        "n_benign": len(benign),
        "seed": args.seed,
        "empty_think_wrap": (
            "assistant turns lacking a leading <think> get an empty think "
            "block so the think-SFT template's generation prompt (which ends "
            "with <think>) is a true prefix; Tier-2 divergence, the EM "
            "control trains no-reasoning responses in a think template"
        ),
        "benign_sampling": (
            "seeded permutation walk keeping rows that pass the trainer's "
            "template-prefix guard (01_build sample-then-filter pattern)"
        ),
        "benign_rejected_during_walk": rejected,
        "n_total": len(rows),
    }
    (args.out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"EM_BUILD_DONE wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
