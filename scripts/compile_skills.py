#!/usr/bin/env python3
"""Compile installed skills into compact cached checklists."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from harness.config import Settings  # noqa: E402
from harness.skills import SkillCompiler  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("skills", nargs="*")
    ap.add_argument("--skills-dir", default="~/.codex/skills")
    ap.add_argument("--cache-dir", default="~/.ai-harness/compiled-skills")
    ap.add_argument("--model-class", default="default")
    args = ap.parse_args()
    settings = Settings()
    settings.skills.enabled = True
    settings.skills.dir = args.skills_dir
    settings.skills.cache_dir = args.cache_dir
    compiler = SkillCompiler(settings, args.model_class)
    names = args.skills or [p.name for p in Path(args.skills_dir).expanduser().iterdir() if p.is_dir()]
    for name in names:
        compiled = compiler.compile(name)
        print(f"{name}: {'compiled' if compiled else 'missing'}")


if __name__ == "__main__":
    main()
