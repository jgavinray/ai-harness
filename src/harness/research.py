"""Codified research pipeline.

Explicit `research: <source>` requests are fetched, chunked, summarized by a
cheap backend, cached by query hash, and injected as a framed brief. Raw
retrieval never enters the main conversation.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from harness.backends.pool import BackendPool, PooledBackend
from harness.config import Settings
from harness.ir import Conversation, TextPart
from harness.research_io import chunks, fetch_source, research_backend, summarize

HEADER = "## Research brief"


class ResearchManager:
    def __init__(self, settings: Settings) -> None:
        self.cfg = settings.research
        self.cache = Path(self.cfg.cache_dir).expanduser()

    async def ensure(self, conv: Conversation, pool: BackendPool, metrics: dict) -> str | None:
        query = _query(conv)
        if not self.cfg.enabled or not query:
            return None
        cached = self._read(query)
        if cached:
            metrics["research_cached"] = 1
            return cached
        source = await fetch_source(query, self.cfg.max_chars)
        if not source:
            return None
        backend = research_backend(pool)
        summaries = []
        for chunk in chunks(source, self.cfg.chunk_chars):
            summaries.append(await summarize(backend, query, chunk))
        brief = "\n".join(s for s in summaries if s).strip()
        if not brief:
            return None
        self._write(query, brief)
        metrics["research_briefs"] = 1
        return brief

    def inject(self, conv: Conversation, brief: str | None) -> Conversation:
        if not brief or HEADER in conv.system:
            return conv
        return replace(conv, system=f"{conv.system}\n\n{HEADER}\n{brief}")

    def _path(self, query: str) -> Path:
        return self.cache / f"{query_hash(query)}.md"

    def _read(self, query: str) -> str:
        path = self._path(query)
        return path.read_text() if path.exists() else ""

    def _write(self, query: str, brief: str) -> None:
        self.cache.mkdir(parents=True, exist_ok=True)
        self._path(query).write_text(brief)


def _query(conv: Conversation) -> str:
    for turn in reversed(conv.turns):
        if turn.role != "user":
            continue
        text = "\n".join(p.text for p in turn.parts if isinstance(p, TextPart))
        marker = "research:"
        idx = text.lower().find(marker)
        if idx != -1:
            return text[idx + len(marker):].strip().splitlines()[0]
    return ""


def query_hash(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()[:24]


def memory_fact(conv: Conversation, brief: str) -> str | None:
    query = _query(conv)
    if not query or not brief:
        return None
    first = next((line.strip("- ") for line in brief.splitlines() if line.strip()), "")
    if not first:
        return None
    return f"- research {query_hash(query)}: {first[:180]}"

