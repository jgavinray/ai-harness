"""Routes each request to a fleet backend.

Priority: session affinity (keeps a conversation's prefix hot in one
server's KV cache) > role match (haiku-class -> fast) > least in-flight,
with main->subagent overflow and circuit-breaker skipping.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from harness.backends.pool import BackendPool, PooledBackend
from harness.config import Settings

AFFINITY_TTL_S = 3600.0
KEY_BASIS_CHARS = 2048


def _flatten(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def session_key(body: dict) -> str:
    """Stable per-conversation key: system prompt + first user message.

    Both are constant across the turns of one Claude Code session, so every
    turn routes to the same backend and rides its KV prefix cache.
    """
    system = _flatten(body.get("system") or "")
    first_user = ""
    for msg in body.get("messages") or []:
        if msg.get("role") == "user":
            first_user = _flatten(msg.get("content"))
            break
    basis = (system + "\x00" + first_user)[:KEY_BASIS_CHARS]
    return hashlib.sha1(basis.encode()).hexdigest()


class Router:
    def __init__(self, pool: BackendPool, settings: Settings) -> None:
        self.pool = pool
        self.settings = settings
        self.affinity: dict[str, tuple[str, float]] = {}

    def pick(self, body: dict) -> PooledBackend:
        key = session_key(body)
        hit = self.affinity.get(key)
        if hit:
            name, ts = hit
            backend = self.pool.get(name)
            if backend and not backend.is_down() and time.time() - ts < AFFINITY_TTL_S:
                self.affinity[key] = (name, time.time())
                return backend

        role = "fast" if "haiku" in (body.get("model") or "") else "main"
        candidates = self.pool.with_role(role)
        if role == "main":
            # overflow: if every main backend is busy (or none up), widen to subagent
            if not candidates or min(b.in_flight for b in candidates) > 0:
                candidates = candidates + self.pool.with_role("subagent")
        if not candidates:
            # everything for the role is down; last resort = any backend at all
            candidates = [b for b in self.pool.backends if not b.is_down()] or self.pool.backends

        chosen = min(candidates, key=lambda b: b.in_flight)
        self.affinity[key] = (chosen.name, time.time())
        return chosen
