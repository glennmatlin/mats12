"""Replication gate, part 2: speedrun LoRA SFT (anchor-post recipe, reconstructed).

Pinned (Tier 1): LoRA rank 64, all 32 layers, MLP + attention, 1 epoch on a 25K
stratified Dolci-Think split, mid-train base checkpoint.
Reconstructed (Tier 2, RECONSTRUCTION.md): LR 1e-4 cosine 3% warmup, effective
batch 32 (micro 1 x accum 32), seq cap 8192, alpha 16 / dropout 0.1
(open-instruct Tulu LoRA precedent), loss on the final assistant turn only.

Loss masking: prompt = chat template over messages[:-1] with generation prompt
(which opens `<think>`); completion = the remainder of the templated string.
Prompt and completion are tokenized separately and concatenated, so the mask
boundary is exact by construction (open-instruct style).

--filter-manifest (Step-4 retrains): JSON {"drop_ids": [...]} of split doc ids
to exclude before training; "none" trains on the full split.
"""

import argparse
import json
import time
from pathlib import Path

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True, help="think-SFT tokenizer")
    parser.add_argument("--split", type=Path, required=True, help="split.parquet from 01_build")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--filter-manifest", default="none")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument(
        "--full-ft",
        action="store_true",
        help="R1 regime: full fine-tune, Adafactor (no bnb in the venv), one epoch; "
        "saves full weights to <out-dir>/model instead of a PEFT adapter",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="allow reusing an out-dir with an existing adapter"
    )
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)

    ds = load_dataset("parquet", data_files=str(args.split), split="train")
    n_split = len(ds)
    if args.filter_manifest != "none":
        drop = set(json.loads(Path(args.filter_manifest).read_text())["drop_ids"])
        ds = ds.filter(lambda r: r["id"] not in drop)
        print(f"filter manifest dropped {n_split - len(ds)} of {n_split} rows")
    print(f"training rows: {len(ds)}")

    def encode(row: dict) -> dict:
        msgs = row["messages"]
        full = tok.apply_chat_template(msgs, tokenize=False)
        prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=False)
        if not full.startswith(prompt):
            raise ValueError(f"template prefix mismatch for id {row['id']!r}")
        prompt_ids = tok(prompt, add_special_tokens=False).input_ids
        completion_ids = tok(full[len(prompt) :], add_special_tokens=False).input_ids
        input_ids = (prompt_ids + completion_ids)[: args.max_len]
        labels = ([-100] * len(prompt_ids) + completion_ids)[: args.max_len]
        return {"input_ids": input_ids, "labels": labels}

    encoded = ds.map(encode, remove_columns=ds.column_names, num_proc=4, desc="tokenize")
    encoded = encoded.filter(lambda r: sum(x != -100 for x in r["labels"]) > 0)
    print(f"encoded rows with completion tokens: {len(encoded)}")

    # transformers v5 dropped warmup_ratio; derive warmup_steps from the ratio.
    steps_per_epoch = -(-len(encoded) // (args.micro_batch * args.grad_accum))
    warmup_steps = max(1, round(args.warmup_ratio * steps_per_epoch * args.epochs))
    print(f"steps/epoch {steps_per_epoch}, warmup steps {warmup_steps}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    if args.full_ft:
        model.gradient_checkpointing_enable()
    else:
        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    save_name = "model" if args.full_ft else "adapter"
    if (args.out_dir / save_name).exists() and not args.overwrite:
        raise SystemExit(
            f"{args.out_dir}/{save_name} already exists; downstream evals and the frozen "
            "subspace recompute reference these weights - pass --overwrite to replace"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.out_dir / "trainer"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.micro_batch,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            bf16=True,
            tf32=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=10,
            save_strategy="no",
            seed=args.seed,
            data_seed=args.seed,
            report_to=[],
            label_names=["labels"],
            optim="adafactor" if args.full_ft else "adamw_torch",
        ),
        train_dataset=encoded,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )
    result = trainer.train()

    adapter_dir = args.out_dir / save_name
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    (args.out_dir / "trainer_log.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    summary = {
        "n_train": len(encoded),
        "hyperparameters": {
            k: getattr(args, k)
            for k in (
                "lr",
                "warmup_ratio",
                "epochs",
                "micro_batch",
                "grad_accum",
                "lora_r",
                "lora_alpha",
                "lora_dropout",
                "max_len",
                "seed",
            )
        },
        "filter_manifest": args.filter_manifest,
        "train_loss": result.training_loss,
        "runtime_s": round(time.time() - t0, 1),
    }
    (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"TRAIN_SPEEDRUN_DONE adapter at {adapter_dir} loss {result.training_loss:.4f}")


if __name__ == "__main__":
    main()
