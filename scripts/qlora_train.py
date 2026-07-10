#!/usr/bin/env python3
"""QLoRA fine-tune of the gated corpus on a CUDA host (flywheel phase 2).

The corpus rows are {"messages": [...]} chat records (trl's conversational
format). Heavy imports live inside train() so --dry-run works on any host —
the flywheel emits this command from a box with no GPU stack installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def resolve(args: argparse.Namespace) -> dict:
    rows = sum(1 for line in Path(args.data).open() if line.strip())
    return {
        "model": args.model,
        "data": args.data,
        "out": args.out,
        "rows": rows,
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_seq_len": args.max_seq_len,
        "quant": args.quant,
    }


def normalize(row: dict) -> dict:
    """Corpus rows carry OpenAI wire format: tool_calls[].function.arguments
    is a JSON string. Qwen's chat template iterates arguments as a mapping,
    so parse the strings (unparseable ones wrap as {"raw": ...})."""
    for message in row["messages"]:
        for call in message.get("tool_calls") or []:
            arguments = call.get("function", {}).get("arguments")
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                except ValueError:
                    parsed = None
                if not isinstance(parsed, dict):
                    parsed = {"raw": arguments}
                call["function"]["arguments"] = parsed
    return row


def check_data(cfg: dict) -> None:
    """Tokenizer-only preflight: render every normalized row through the
    model's chat template so format mismatches surface all at once, without
    paying the 52 GB model load per failure."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    failures = 0
    for i, line in enumerate(Path(cfg["data"]).open()):
        if not line.strip():
            continue
        row = normalize(json.loads(line))
        try:
            tokenizer.apply_chat_template(row["messages"], tokenize=False)
        except Exception as exc:
            failures += 1
            if failures <= 5:
                print(f"row {i}: {type(exc).__name__}: {str(exc)[:160]}")
    print(f"checked rows; failures: {failures}")
    if failures:
        raise SystemExit(1)


def train(cfg: dict) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    # --quant none = plain bf16 LoRA: bitsandbytes ships no CUDA aarch64
    # binary (DGX Spark GB10), and its 121 GB unified memory holds bf16.
    load_kwargs: dict = {"device_map": "auto", "dtype": torch.bfloat16}
    if cfg["quant"] == "nf4":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    dataset = load_dataset("json", data_files=cfg["data"], split="train")
    dataset = dataset.map(normalize)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        peft_config=LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
        args=SFTConfig(
            output_dir=cfg["out"],
            num_train_epochs=cfg["epochs"],
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["grad_accum"],
            max_length=cfg["max_seq_len"],
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=10,
            save_strategy="epoch",
        ),
    )
    trainer.train()
    trainer.save_model(cfg["out"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--quant", choices=["nf4", "none"], default="nf4")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-data", action="store_true")
    args = ap.parse_args()
    cfg = resolve(args)
    if args.dry_run:
        print(json.dumps(cfg))
        return
    if args.check_data:
        check_data(cfg)
        return
    train(cfg)


if __name__ == "__main__":
    main()
