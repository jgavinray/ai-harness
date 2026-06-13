#!/usr/bin/env python3
"""Offline project-memory distiller over trace JSONL.

This deterministic pass writes durable, low-risk facts from clean sessions into
the same memory files read by MemoryStage. LLM summarization can layer on later;
this script is intentionally cheap and reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from harness.config import Settings  # noqa: E402
from harness.memory import MemoryManager, project_key  # noqa: E402


def _clean(metrics: dict) -> bool:
    return not any(metrics.get(k) for k in ("invalid_calls", "retries", "degenerate_aborts"))


def _system_text(payload: dict) -> str:
    for msg in payload.get("messages") or []:
        if msg.get("role") == "system":
            return msg.get("content") or ""
    return ""


def _tool_facts(row: dict) -> list[str]:
    facts: list[str] = []
    for ev in row.get("events") or []:
        if ev.get("t") != "tool_call" or ev.get("name") != "Bash":
            continue
        cmd = (ev.get("arguments") or {}).get("command")
        if cmd and len(cmd) <= 120:
            facts.append(f"- verified command: `{cmd}`")
    return facts


def _facts(row: dict) -> tuple[str, list[str]]:
    payload = row.get("payload") or {}
    system = _system_text(payload)
    pkey = project_key(system)
    facts = []
    if pkey != "default":
        facts.append(f"- project key: {pkey}")
    facts.extend(_tool_facts(row))
    return pkey, facts


def distill(trace_path: Path, settings: Settings) -> tuple[int, int]:
    manager = MemoryManager(settings, None)
    projects: dict[str, list[str]] = {}
    total = 0
    for line in trace_path.read_text().splitlines():
        if not line.strip():
            continue
        total += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not _clean(row.get("metrics") or {}):
            continue
        pkey, facts = _facts(row)
        projects.setdefault(pkey, []).extend(facts)
    for pkey, facts in projects.items():
        manager.merge(pkey, "\n".join(facts))
    return len(projects), total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="traces/sessions.jsonl")
    ap.add_argument("--memory-dir", default="~/.ai-harness/memory")
    ap.add_argument("--max-chars", type=int, default=4000)
    args = ap.parse_args()
    settings = Settings()
    settings.memory.dir = args.memory_dir
    settings.memory.max_chars = args.max_chars
    projects, total = distill(Path(args.traces), settings)
    print(f"distilled {projects} projects from {total} trace rows")


if __name__ == "__main__":
    main()
