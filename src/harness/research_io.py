"""Fetch and summarize source material for the research pipeline."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import httpx

from harness.backends.pool import BackendPool, PooledBackend
from harness.ir import TextDelta


async def fetch_source(query: str, max_chars: int) -> str:
    parsed = urlparse(query)
    if parsed.scheme == "file":
        return Path(parsed.path).read_text()[:max_chars]
    if parsed.scheme in ("http", "https"):
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(query)
            resp.raise_for_status()
            return resp.text[:max_chars]
    return query[:max_chars]


def chunks(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or [text]


def research_backend(pool: BackendPool) -> PooledBackend:
    candidates = pool.with_role("research") or pool.with_role("fast") or pool.backends
    return min(candidates, key=lambda b: (b.in_flight, b.requests))


async def summarize(backend: PooledBackend, query: str, chunk: str) -> str:
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
