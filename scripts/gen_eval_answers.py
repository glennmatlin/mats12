"""Generate model answers for eval questions (runs on PACE GPU).

Base condition (default): the raw question text, no chat template — the
mid-train base has no chat format, and the post prompts it directly (its
"loses focus" behavior is expected and handled by the slop-aware judge).
Generation settings are the frozen Tier-2 choices (RECONSTRUCTION.md):
temp 0.8, top-p 0.95, 512 new tokens.

SFT condition (--chat-template): the think-SFT chat template wraps the
question; the generation prompt opens `<think>`, so completions are
"reasoning</think>answer". The post-think answer is stored as `completion`
(what the judge scores) and the full text as `completion_raw`; a completion
that never closes its think block gets think_truncated=true and an empty
answer, which judging skips and counts. Divergence-ledger entry 2026-08-14
covers the SFT-condition settings (chat template, 2048 max tokens, eos ids
from the template's <|im_end|>/eos pair).

Prefill condition (--prefill "Okay,"): base condition with the answer opener
appended after a blank line, reconstructing the anchor post's Okay-prefill
elicitation probe. The stored `completion` includes the prefill text (the
judge scores the full visible answer); `prefill` and `completion_generated`
record the split. Base condition only — the SFT chat template has its own
answer scaffold. Recovery criterion pre-committed in RECONSTRUCTION.md
2026-08-16 before any prefill job runs.

Output: <out-dir>/<behavior>.jsonl with {question_id, question, completion, ...}.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None, help="default: model path")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--prefill", default="", help="answer opener, e.g. 'Okay,'; base only")
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--behaviors", default="", help="csv filter; empty = all")
    parser.add_argument("--n-per-behavior", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--force", action="store_true", help="overwrite existing output jsonl")
    args = parser.parse_args()
    if args.prefill and args.chat_template:
        parser.error("--prefill is a base-condition probe; incompatible with --chat-template")

    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cuda"
    )
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    gen_kwargs: dict = {}
    if args.chat_template:
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        eos_ids = [i for i in {im_end, tok.eos_token_id} if i is not None]
        gen_kwargs["eos_token_id"] = eos_ids
        print(f"chat mode: eos ids {eos_ids}")

    def build_inputs(question: str):
        if args.chat_template:
            text = tok.apply_chat_template(
                [{"role": "user", "content": question}],
                add_generation_prompt=True,
                tokenize=False,
            )
            return tok(text, add_special_tokens=False, return_tensors="pt").to("cuda")
        if args.prefill:
            return tok(f"{question}\n\n{args.prefill}", return_tensors="pt").to("cuda")
        return tok(question, return_tensors="pt").to("cuda")

    behavior_files = sorted(
        p
        for p in args.eval_dir.glob("*.json")
        if p.stem not in {"seeds", "manifest", "curation_v2"}
    )
    if args.behaviors:
        wanted = set(args.behaviors.split(","))
        behavior_files = [p for p in behavior_files if p.stem in wanted]
    for path in behavior_files:
        questions = json.loads(path.read_text())
        rng = random.Random(args.seed)
        idxs = rng.sample(range(len(questions)), min(args.n_per_behavior, len(questions)))
        out_path = args.out_dir / f"{path.stem}.jsonl"
        if out_path.exists() and not args.force:
            raise SystemExit(
                f"{out_path} exists; generations are nondeterministic and judged "
                "artifacts may reference these bytes - pass --force to overwrite"
            )
        with out_path.open("w") as f:
            for i in idxs:
                inputs = build_inputs(questions[i])
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        **gen_kwargs,
                    )
                completion = tok.decode(
                    out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
                record = {
                    "question_id": f"{path.stem}-{i}",
                    "question": questions[i],
                    "completion": completion,
                }
                if args.prefill:
                    record.update(
                        completion=f"{args.prefill}{completion}",
                        completion_generated=completion,
                        prefill=args.prefill,
                    )
                if args.chat_template:
                    if "</think>" in completion:
                        answer = completion.split("</think>", 1)[1].strip()
                        record.update(
                            completion=answer, completion_raw=completion, think_truncated=False
                        )
                    else:
                        record.update(
                            completion="", completion_raw=completion, think_truncated=True
                        )
                f.write(json.dumps(record) + "\n")
        print(f"{path.stem}: {len(idxs)} generations -> {out_path}")
    print("GEN_EVAL_ANSWERS_DONE")


if __name__ == "__main__":
    main()
