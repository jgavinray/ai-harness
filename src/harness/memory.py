"""Memory v1: per-project lessons that persist across sessions.

The knowledge cache: facts a previous session already derived (build
commands, conventions, gotchas) are injected into the next session's system
prompt instead of being re-explored at 20k tokens a time.

A session is considered finished when it has been idle for idle_s; its
transcript tail is then summarized by a fast-role backend into bullet facts
and merged into the project's memory file. Disabled by default.
"""

from __future__ import annotations

import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Awaitable, Callable

from harness.config import Settings
from harness.ir import Conversation
from harness.tokens.counter import TokenCounter

HEADER = "## Project memory (from previous sessions)"

EXTRACT_PROMPT = """\
You summarize a coding-agent session. From the transcript below, extract up to 6 durable facts \
about this PROJECT that would save time in future sessions: build/test/lint commands that worked, \
project conventions, file locations of key things, gotchas that caused failures. \
Only include facts likely to stay true. Output one fact per line, each starting with "- ". \
If there is nothing durable, output nothing.

Transcript:
{transcript}
"""

Completer = Callable[[list[dict]], Awaitable[str]]


def project_key(system: str) -> str:
    m = re.search(r"Working directory:\s*(\S+)", system)
    if not m:
        return "default"
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", m.group(1)).strip("-") or "default"


def _transcript_tail(messages: list[dict], max_chars: int = 8000) -> str:
    parts = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str) and content:
            parts.append(f"{msg.get('role')}: {content[:600]}")
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            parts.append(f"assistant calls {fn.get('name')} {fn.get('arguments', '')[:200]}")
    return "\n".join(parts)[-max_chars:]


class MemoryManager:
    def __init__(self, settings: Settings, completer: Completer | None) -> None:
        self.cfg = settings.memory
        self.completer = completer
        self.dir = Path(self.cfg.dir).expanduser()
        self.sessions: dict[str, tuple[float, str, list[dict]]] = {}

    def path_for(self, pkey: str) -> Path:
        return self.dir / f"{pkey}.md"

    def read(self, pkey: str) -> str:
        p = self.path_for(pkey)
        return p.read_text() if p.exists() else ""

    def merge(self, pkey: str, new_text: str) -> None:
        existing = self.read(pkey)
        have = set(existing.splitlines())
        fresh = [
            line.strip()
            for line in new_text.splitlines()
            if line.strip().startswith("- ") and line.strip() not in have
        ]
        if not fresh:
            return
        combined = (existing.rstrip() + "\n" if existing else "") + "\n".join(fresh) + "\n"
        if len(combined) > self.cfg.max_chars:  # keep the newest facts
            combined = combined[-self.cfg.max_chars:]
            combined = combined[combined.find("- "):]
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path_for(pkey).write_text(combined)

    def note(self, session_key: str, system: str, messages: list[dict]) -> None:
        self.sessions[session_key] = (time.time(), project_key(system), messages)

    async def sweep(self, now: float | None = None) -> None:
        if self.completer is None:
            return
        now = now if now is not None else time.time()
        expired = [k for k, (ts, _, _) in self.sessions.items() if now - ts >= self.cfg.idle_s]
        for key in expired:
            _, pkey, messages = self.sessions.pop(key)
            prompt = EXTRACT_PROMPT.format(transcript=_transcript_tail(messages))
            try:
                facts = await self.completer([{"role": "user", "content": prompt}])
            except Exception:
                continue  # extraction is best-effort; never disturb serving
            if facts:
                self.merge(pkey, facts)


class MemoryStage:
    def __init__(self, manager: MemoryManager, settings: Settings) -> None:
        self.manager = manager
        self.settings = settings

    def apply(self, conv: Conversation, settings: Settings) -> Conversation:
        if not settings.memory.enabled or HEADER in conv.system:
            return conv
        mem = self.manager.read(project_key(conv.system))
        if not mem:
            return conv
        return replace(conv, system=f"{conv.system}\n\n{HEADER}\n{mem}")


def injected_memory_tokens(system: str, counter: TokenCounter) -> int:
    if HEADER not in system:
        return 0
    return counter.count_text(system.split(HEADER, 1)[1])
