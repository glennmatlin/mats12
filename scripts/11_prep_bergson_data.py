"""Prepare Bergson trackstar inputs for the escalated behaviors.

Emits (a) the 25K training split as jsonl with the raw `messages` column
(bergson conversation_column path: chat-templated, loss masked to assistant
spans — Tier-2 divergence from training's last-turn-only loss for the 588
multi-assistant docs, recorded in RECONSTRUCTION.md), and (b) per-behavior
query jsonl {question_id, prompt, completion}: the frozen eval question as
the user turn, the Step-1 SFT-condition generation as the assistant turn.
Both derive only from frozen artifacts.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True, help="split.parquet")
    parser.add_argument("--gens-dir", type=Path, required=True, help="Step-1 SFT gens dir")
    parser.add_argument("--behaviors", default="both_sides,china_friendly")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="when set, drop docs whose assistant text the chat template alters "
        "(bergson's span search requires verbatim recovery); dropped docs are "
        "unscoreable and thus never filtered",
    )
    args = parser.parse_args()
    import pyarrow.parquet as pq

    args.out_dir.mkdir(parents=True, exist_ok=True)

    tok = None
    if args.tokenizer is not None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def template_compatible(messages: list[dict]) -> bool:
        if tok is None:
            return True
        s = tok.apply_chat_template(messages, tokenize=False)
        pos = 0
        for m in messages:
            if m["role"] != "assistant":
                continue
            start = s.rfind(m["content"], pos)
            if start < 0:
                return False
            pos = start + len(m["content"])
        return True

    corpus_path = args.out_dir / "corpus.jsonl"
    table = pq.read_table(args.split, columns=["id", "messages"])
    kept = dropped = 0
    with corpus_path.open("w") as f:
        for doc_id, messages in zip(
            table.column("id").to_pylist(), table.column("messages").to_pylist(), strict=True
        ):
            if not template_compatible(messages):
                dropped += 1
                print(f"dropped (template-incompatible): {doc_id}")
                continue
            kept += 1
            f.write(json.dumps({"id": str(doc_id), "messages": messages}) + "\n")
    print(f"corpus: {kept} docs kept, {dropped} dropped -> {corpus_path}")

    for behavior in args.behaviors.split(","):
        rows = [
            json.loads(line)
            for line in (args.gens_dir / f"{behavior}.jsonl").read_text().splitlines()
            if line
        ]
        out = args.out_dir / f"queries_{behavior}.jsonl"
        with out.open("w") as f:
            for r in rows:
                f.write(
                    json.dumps(
                        {
                            "question_id": r["question_id"],
                            "prompt": r["question"],
                            "completion": r["completion"],
                        }
                    )
                    + "\n"
                )
        print(f"{behavior}: {len(rows)} queries -> {out}")


if __name__ == "__main__":
    main()
