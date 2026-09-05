"""Replication gate, part 1: rebuild the anchor post's speedrun SFT sample.

Recipe (post appendix + Fig. 'Speedrun SFT is a stratified sub-sample'):
- filter Dolci-Think-SFT examples > 8192 tokens (2.27M -> ~1.19M)
- stratified sample preserving the pinned dataset-source proportions
- 5 splits x 25K in the post; TDA ran on one 25K split, so we build split 0

Implementation notes (Tier-2, disclosed):
- Token length = chat-templated full conversation (think-SFT tokenizer,
  add_special_tokens=False), i.e. the training tokenization. Template adds
  ~40 tokens of system-prompt overhead vs raw content.
- Sample-then-filter per stratum: walk a seeded permutation of each stratum,
  keep rows <= max_tokens until the quota fills. Distribution-identical to
  filter-then-sample within stratum, ~50x cheaper than tokenizing 2.27M rows.
  Per-stratum keep-rates are logged; the post's global keep rate (~52%) is a
  recipe sanity anchor.
- "other" and the merged wildjailbreak+wildguardmix stratum pool their
  sub-sources and sample uniformly (proportional to post-filter availability).
"""

import argparse
import collections
import json
import random
from pathlib import Path

STRATA: dict[str, float] = {
    "correct-python-sft": 0.336,
    "persona-precise-if": 0.186,
    "if_qwq_reasoning_verified": 0.103,
    "aya-100k": 0.082,
    "synthetic-2-sft-chinese": 0.050,
    "openthoughts3-science": 0.043,
    "tulu-wildchat": 0.036,
    "wildjailbreak-wildguardmix": 0.067,
    "other": 0.097,
}

# Exact source -> stratum mapping for the 16 dataset_source values present in
# allenai/Dolci-Think-SFT-7B (enumerated on PACE 2026-08-14, 2,268,178 rows).
# An unknown source raises: fail fast rather than silently mis-stratify.
SOURCE_TO_STRATUM: dict[str, str] = {
    "saumyamalik/correct-python-sft-187k-x16-thoughts-filtered-decontam-v2": "correct-python-sft",
    "allenai/persona-precise-if-r1-final-content-filtered-chinese-filtered": "persona-precise-if",
    "saumyamalik/if_qwq_reasoning_verified_filtered_decontam_v2": "if_qwq_reasoning_verified",
    "allenai/aya-100k-r1-format-filtered-keyword-filtered-filter-datecutoff-ngram-filtered": "aya-100k",  # noqa: E501
    "allenai/SYNTHETIC-2-SFT-cn-fltrd-final-ngram-filtered-chinese-filtered": "synthetic-2-sft-chinese",  # noqa: E501
    "saumyamalik/OpenThoughts3-full-filtered-science-decontam-v2": "openthoughts3-science",
    "allenai/tulu_v3.9_wildchat_100k_english-r1-final-content-filtered": "tulu-wildchat",
    "allenai/wildjailbreak-r1-v2-format-filtered-keyword-filtered-filter-datecutoff-final-content-filtered": "wildjailbreak-wildguardmix",  # noqa: E501
    "allenai/wildguardmix-r1-v2-all-filtered-ngram-filtered-chinese-filtered": "wildjailbreak-wildguardmix",  # noqa: E501
    "saumyamalik/OpenThoughts3-full-filtered-math-decontam-v2": "other",
    "saumyamalik/OpenThoughts3-full-filtered-code-subsampled-decontam-v2": "other",
    "allenai/nemotron-post-training-dataset-subset-ngram-filtered-no-tool-calls": "other",
    "allenai/wildchat-r1-p2-repetition-filter": "other",
    "allenai/coconot-r1-format-domain-filtered-keyword-filtered-filter-datecutoff-chinese-filtered": "other",  # noqa: E501
    "allenai/oasst1-r1-format-filtered-keyword-filtered-filter-datecutoff-chinese-filtered": "other",  # noqa: E501
    "allenai/tablegpt_r1-format-filtered-keyword-filtered-filter-datecutoff": "other",
}


def largest_remainder_quotas(proportions: dict[str, float], total: int) -> dict[str, int]:
    raw = {k: v * total for k, v in proportions.items()}
    quotas = {k: int(v) for k, v in raw.items()}
    short = total - sum(quotas.values())
    for k in sorted(raw, key=lambda k: raw[k] - quotas[k], reverse=True)[:short]:
        quotas[k] += 1
    return quotas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="dir of Dolci parquet shards")
    parser.add_argument("--tokenizer-path", type=Path, required=True, help="think-SFT tokenizer")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split-size", type=int, default=25_000)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    shards = sorted(args.data_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {args.data_dir}")

    # Pass 1: locate every row by stratum (columns-only read, cheap).
    locations: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    source_counts: collections.Counter[str] = collections.Counter()
    for si, shard in enumerate(shards):
        col = pq.read_table(shard, columns=["dataset_source"]).column("dataset_source").to_pylist()
        for ri, src in enumerate(col):
            if src not in SOURCE_TO_STRATUM:
                raise ValueError(f"unmapped dataset_source {src!r} in {shard.name}")
            locations[SOURCE_TO_STRATUM[src]].append((si, ri))
        source_counts.update(col)
    total_rows = sum(source_counts.values())
    print(f"pass 1: {total_rows} rows across {len(shards)} shards")

    quotas = largest_remainder_quotas(STRATA, args.split_size)
    for stratum, quota in quotas.items():
        pool = len(locations[stratum])
        print(f"  {stratum}: quota {quota}, raw pool {pool}")
        if pool < quota:
            raise ValueError(f"stratum {stratum!r}: pool {pool} < quota {quota}")

    def token_lengths(msgs_batch: list[list[dict]]) -> list[int]:
        texts = [tok.apply_chat_template(m, tokenize=False) for m in msgs_batch]
        return [len(ids) for ids in tok(texts, add_special_tokens=False).input_ids]

    kept_rows: list[dict] = []
    stratum_stats: dict[str, dict] = {}
    for stratum, quota in quotas.items():
        rng = random.Random(f"{args.seed}:{stratum}")
        perm = list(locations[stratum])
        rng.shuffle(perm)
        kept: list[dict] = []
        scanned = 0
        cursor = 0
        while len(kept) < quota and cursor < len(perm):
            chunk = perm[cursor : cursor + max(1024, int((quota - len(kept)) * 1.5))]
            cursor += len(chunk)
            by_shard: dict[int, list[int]] = collections.defaultdict(list)
            for si, ri in chunk:
                by_shard[si].append(ri)
            fetched: dict[tuple[int, int], dict] = {}
            for si, ris in by_shard.items():
                table = pq.read_table(shards[si], columns=["id", "dataset_source", "messages"])
                for ri, row in zip(ris, table.take(ris).to_pylist(), strict=True):
                    fetched[(si, ri)] = row
            rows = [fetched[loc] for loc in chunk]
            lengths = token_lengths([r["messages"] for r in rows])
            for row, n_tokens in zip(rows, lengths, strict=True):
                scanned += 1
                if n_tokens <= args.max_tokens:
                    row["n_tokens"] = n_tokens
                    kept.append(row)
                    if len(kept) == quota:
                        break
        if len(kept) < quota:
            raise ValueError(
                f"stratum {stratum!r}: only {len(kept)}/{quota} rows pass the "
                f"{args.max_tokens}-token filter (scanned {scanned})"
            )
        keep_rate = len(kept) / scanned if scanned else 0.0
        stratum_stats[stratum] = {
            "quota": quota,
            "scanned": scanned,
            "keep_rate_estimate": round(keep_rate, 4),
        }
        print(f"  {stratum}: filled {quota} (scanned {scanned}, keep rate ~{keep_rate:.2%})")
        kept_rows.extend(kept)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = args.out_dir / "split.parquet"
    pq.write_table(pa.Table.from_pylist(kept_rows), out_parquet)

    weighted_keep = sum(STRATA[s] * st["keep_rate_estimate"] for s, st in stratum_stats.items())
    manifest = {
        "seed": args.seed,
        "split_size": args.split_size,
        "max_tokens": args.max_tokens,
        "tokenizer": str(args.tokenizer_path),
        "total_pool_rows": total_rows,
        "strata_proportions": STRATA,
        "source_to_stratum": SOURCE_TO_STRATUM,
        "source_counts": dict(source_counts),
        "stratum_stats": stratum_stats,
        "weighted_keep_rate_estimate": round(weighted_keep, 4),
        "post_keep_rate_anchor": "2.27M -> ~1.19M (~52%); sample-weighted, so approximate",
        "token_count_mean": round(sum(r["n_tokens"] for r in kept_rows) / len(kept_rows), 1),
        "ids": [r["id"] for r in kept_rows],
    }
    (args.out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"BUILD_SPLIT_DONE wrote {out_parquet} ({len(kept_rows)} rows)")


if __name__ == "__main__":
    main()
