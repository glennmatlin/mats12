"""Generate +5/-5 synthetic rubric examples for the probe scorer (32B, PACE GPU).

The primary scorer's training data (PREREGISTRATION Setting, 2026-08-16):
for each behavior and each frozen eval question, allenai/Olmo-3-32B-Think
writes one response targeted to score +5 and one to score -5 under the
behavior's rubric. Tier-2 pins (RECONSTRUCTION ledger 2026-08-16): chat
template, temperature 0.7, max 2048 new tokens, one sample per
(behavior, question, target); the post-</think> answer is the example
(think-truncated rows flagged and excluded downstream); judge-QC filter
applied after generation, before probe fit.

Output: <out-dir>/<behavior>.jsonl with
{question_id, target, completion, completion_raw, think_truncated}.
"""

import argparse
import json
import random
import time
from pathlib import Path

from mats12.behaviors import BEHAVIORS, RubricStatus

PROMPT_TEMPLATE = (
    "Write a response to the question below that would score {target} under "
    "this scoring rubric.\n\nRubric:\n{rubric}\n\nQuestion: {question}\n\n"
    "Give only the response itself."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--behaviors", default="", help="csv filter; empty = all judgeable")
    parser.add_argument("--n-per-behavior", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260813, help="must match the eval runs")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch-gen", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="overwrite existing output jsonl")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [i for i in {im_end, tok.eos_token_id} if i is not None]

    wanted = set(args.behaviors.split(",")) if args.behaviors else None
    behavior_files = sorted(
        p
        for p in args.eval_dir.glob("*.json")
        if p.stem in BEHAVIORS and (wanted is None or p.stem in wanted)
    )
    for path in behavior_files:
        behavior = BEHAVIORS[path.stem]
        if behavior.rubric_status is RubricStatus.SUMMARY:
            raise SystemExit(f"behavior '{path.stem}' has a placeholder rubric; refusing")
        questions = json.loads(path.read_text())
        rng = random.Random(args.seed)
        idxs = rng.sample(range(len(questions)), min(args.n_per_behavior, len(questions)))
        out_path = args.out_dir / f"{path.stem}.jsonl"
        if out_path.exists() and not args.force:
            raise SystemExit(f"{out_path} exists; pass --force to overwrite")

        jobs = [(f"{path.stem}-{i}", questions[i], target) for i in idxs for target in ("+5", "-5")]
        t0 = time.time()
        with out_path.open("w") as f:
            for start in range(0, len(jobs), args.batch_gen):
                batch = jobs[start : start + args.batch_gen]
                texts = [
                    tok.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": PROMPT_TEMPLATE.format(
                                    target=target, rubric=behavior.rubric, question=q
                                ),
                            }
                        ],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    for _, q, target in batch
                ]
                enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(
                    "cuda"
                )
                with torch.no_grad():
                    gen = model.generate(
                        **enc,
                        do_sample=True,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        min_new_tokens=1,
                        eos_token_id=eos_ids,
                        pad_token_id=tok.pad_token_id,
                    )
                prompt_len = enc["input_ids"].shape[1]
                for (qid, _, target), row in zip(batch, gen, strict=False):
                    raw = tok.decode(row[prompt_len:], skip_special_tokens=True)
                    if "</think>" in raw:
                        record = {
                            "question_id": qid,
                            "target": target,
                            "completion": raw.split("</think>", 1)[1].strip(),
                            "completion_raw": raw,
                            "think_truncated": False,
                        }
                    else:
                        record = {
                            "question_id": qid,
                            "target": target,
                            "completion": "",
                            "completion_raw": raw,
                            "think_truncated": True,
                        }
                    f.write(json.dumps(record) + "\n")
                f.flush()
                done = min(start + args.batch_gen, len(jobs))
                rate = done / max(time.time() - t0, 1e-9)
                eta = (len(jobs) - done) / max(rate, 1e-9) / 60
                print(
                    f"  {path.stem}: {done}/{len(jobs)} ({rate:.2f}/s, eta {eta:.0f} min)",
                    flush=True,
                )
        print(f"{path.stem}: {len(jobs)} synthetics -> {out_path}", flush=True)
    print("GENERATE_PROBE_SYNTHETICS_DONE")


if __name__ == "__main__":
    main()
