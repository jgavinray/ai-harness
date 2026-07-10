#!/usr/bin/env python3
"""LoRA fine-tune job over the gated corpus (mlx or cuda backend)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def command(corpus: str, base_model: str, out_dir: str, backend: str = "mlx") -> list[str]:
    if backend == "cuda":
        trainer = Path(__file__).resolve().parent / "qlora_train.py"
        return [
            sys.executable, str(trainer),
            "--model", base_model,
            "--data", corpus,
            "--out", out_dir,
        ]
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
    ap.add_argument("--backend", choices=["mlx", "cuda"], default="mlx")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    if not Path(args.corpus).exists():
        raise SystemExit(f"missing corpus: {args.corpus}")
    cmd = command(args.corpus, args.base_model, args.out, backend=args.backend)
    if not args.execute:
        print(" ".join(cmd))
        return
    import subprocess
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
