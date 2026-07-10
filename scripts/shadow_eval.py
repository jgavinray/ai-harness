#!/usr/bin/env python3
"""Shadow-eval candidate backends: list the envelope commands, or run them."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from harness.config import load_settings  # noqa: E402


def candidate_commands(config_path: Path, out_dir: str = "evals/results") -> list[str]:
    settings = load_settings(config_path)
    cmds = []
    for b in settings.backends:
        if "candidate" not in b.roles:
            continue
        cmds.append(
            ".venv/bin/python evals/run.py "
            f"--backend-url {b.base_url} --model {b.model} --profile {b.profile} "
            f"--kind {b.kind} --configs full --out {out_dir}/{b.name}"
        )
    return cmds


def run_commands(cmds: list[str], execute: bool) -> list[int]:
    if not execute:
        for cmd in cmds:
            print(cmd)
        return []
    return [subprocess.run(shlex.split(cmd)).returncode for cmd in cmds]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="harness.toml")
    ap.add_argument("--out", default="evals/results")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    rcs = run_commands(candidate_commands(Path(args.config), args.out), args.execute)
    if any(rcs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
