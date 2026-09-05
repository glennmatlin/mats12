"""Practice loop (off-clock rehearsal): tiny LoRA train + generation on ~100 docs.

Generic pipeline rehearsal only — no behavior rubrics, no experiment logic.
Proves: staged model loads, Dolci rows tokenize, LoRA trains, adapter saves,
generation runs. Outputs land in --out-dir (shared storage), including
practice_generations.jsonl for a local judge smoke.
"""

import argparse
import json
from pathlib import Path

PRACTICE_PROMPTS = [
    "Explain the difference between a list and a tuple in Python.",
    "I've been feeling overwhelmed at work lately. Any advice?",
    "Summarize why the sky is blue in two sentences.",
]


def rows_to_text(row: dict) -> str | None:
    msgs = row.get("messages")
    if msgs:
        return "\n".join(f"{m.get('role', '?')}: {m.get('content', '')}" for m in msgs)
    return row.get("text")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-glob", required=True, help="parquet glob for Dolci rows")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-docs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-len", type=int, default=1024)
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("parquet", data_files=args.data_glob, split="train")
    ds = ds.select(range(min(args.n_docs, len(ds))))
    texts = [t for t in (rows_to_text(r) for r in ds) if t]
    print(f"practice docs: {len(texts)}")

    def tokenize(batch: dict) -> dict:
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    import datasets as hfds

    tokenized = hfds.Dataset.from_dict({"text": texts}).map(
        tokenize, batched=True, remove_columns=["text"]
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cuda"
    )
    model = get_peft_model(model, LoraConfig(r=8, target_modules="all-linear"))
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.out_dir / "trainer"),
            max_steps=args.max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            logging_steps=5,
            save_strategy="no",
            bf16=True,
            report_to=[],
        ),
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    trainer.train()
    model.save_pretrained(args.out_dir / "adapter")

    model.eval()
    records = []
    for prompt in PRACTICE_PROMPTS:
        inputs = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=200, do_sample=True, temperature=0.8, top_p=0.95
            )
        completion = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        records.append({"prompt": prompt, "completion": completion})

    gen_path = args.out_dir / "practice_generations.jsonl"
    with gen_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"PRACTICE_LOOP_DONE wrote {gen_path}")


if __name__ == "__main__":
    main()
