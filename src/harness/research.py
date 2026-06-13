"""Codified research pipeline.

Explicit `research: <source>` requests are fetched, chunked, summarized by a
cheap backend, cached by query hash, and injected as a framed brief. Raw
retrieval never enters the main conversation.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

import httpx

from harness.backends.pool import BackendPool, PooledBackend
from harness.config import Settings
from harness.ir import Conversation, TextDelta, TextPart

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
        source = await _fetch(query, self.cfg.max_chars)
        if not source:
            return None
        backend = _research_backend(pool)
        summaries = []
        for chunk in _chunks(source, self.cfg.chunk_chars):
            summaries.append(await _summarize(backend, query, chunk))
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
        digest = hashlib.sha256(query.encode()).hexdigest()[:24]
        return self.cache / f"{digest}.md"

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


async def _fetch(query: str, max_chars: int) -> str:
    parsed = urlparse(query)
    if parsed.scheme == "file":
        return Path(parsed.path).read_text()[:max_chars]
    if parsed.scheme in ("http", "https"):
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(query)
            resp.raise_for_status()
            return resp.text[:max_chars]
    return query[:max_chars]


def _chunks(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or [text]


def _research_backend(pool: BackendPool) -> PooledBackend:
    candidates = pool.with_role("research") or pool.with_role("fast") or pool.backends
    return min(candidates, key=lambda b: (b.in_flight, b.requests))


async def _summarize(backend: PooledBackend, query: str, chunk: str) -> str:
    payload = {
        "model": backend.model_name,
        "messages": [
            {"role": "system", "content": "Summarize this research source for a coding agent. Return concise bullets."},
            {"role": "user", "content": f"Query: {query}\n\nSource chunk:\n{chunk}"},
        ],
        "max_tokens": 500,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    text = ""
    async for ev in backend.profile.parse(backend.stream(payload)):
        if isinstance(ev, TextDelta):
            text += ev.text
    return text.strip()
