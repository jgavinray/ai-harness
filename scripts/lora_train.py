#!/usr/bin/env python3
"""LoRA fine-tune job scaffold over the gated corpus."""

from __future__ import annotations

import argparse
from pathlib import Path


def command(corpus: str, base_model: str, out_dir: str) -> list[str]:
    return [
        "mlx_lm.lora",
        "--model", base_model,
        "--train",
        "--data", corpus,
        "--adapter-path", out_dir,
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="evals/results/corpus.jsonl")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    if not Path(args.corpus).exists():
        raise SystemExit(f"missing corpus: {args.corpus}")
    cmd = command(args.corpus, args.base_model, args.out)
    if not args.execute:
        print(" ".join(cmd))
        return
    import subprocess
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
