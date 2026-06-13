"""Deterministic skill compiler.

Installed skills can be too verbose for small models. This compiler rewrites a
SKILL.md file into a compact imperative checklist and caches it by content hash
and model class. It is intentionally deterministic so it can run offline or
inside relay feedback without adding another model call.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from harness.config import Settings


class SkillCompiler:
    def __init__(self, settings: Settings, model_class: str = "default") -> None:
        self.cfg = settings.skills
        self.model_class = model_class
        self.root = Path(self.cfg.dir).expanduser()
        self.cache = Path(self.cfg.cache_dir).expanduser()

    def compile(self, name: str) -> str | None:
        path = self._find(name)
        if path is None:
            return None
        text = path.read_text()
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        cache_path = self.cache / f"{_safe(name)}-{self.model_class}-{digest}.md"
        if cache_path.exists():
            return cache_path.read_text()
        compiled = _compile_text(text, self.cfg.max_tokens)
        self.cache.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(compiled)
        return compiled

    def _find(self, name: str) -> Path | None:
        safe = _safe(name)
        candidates = [
            self.root / name / "SKILL.md",
            self.root / safe / "SKILL.md",
            self.root / name,
            self.root / safe,
        ]
        for path in candidates:
            if path.is_file():
                return path
        return None


def skill_name(args: dict) -> str:
    return str(args.get("name") or args.get("skill") or args.get("skill_name") or "")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "skill"


def _compile_text(text: str, max_tokens: int) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        if line.startswith("#"):
            continue
        if line.startswith(("-", "*")):
            line = line[1:].strip()
        elif re.match(r"^\d+[.)]\s+", line):
            line = re.sub(r"^\d+[.)]\s+", "", line)
        elif len(line) > 120:
            continue
        if line:
            lines.append(line.rstrip("."))
    if not lines:
        lines = ["Read the skill instructions, follow them in order, and verify the result"]
    limit = max_tokens * 4
    out: list[str] = []
    total = 0
    for line in lines:
        item = f"{len(out) + 1}. {line}"
        if total + len(item) + 1 > limit:
            break
        out.append(item)
        total += len(item) + 1
    return "\n".join(out) + "\n"
