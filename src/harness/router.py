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
# Above this estimated context size, a session never leaves its KV-warm
# backend for capacity reasons: a cold re-prefill (40-120s observed at 75k
# tokens) costs far more than briefly queuing on the warm slot.
STICKY_CONTEXT_TOKENS = 8192


# Claude Code prepends a per-request billing block; its content varies between
# requests of one session, so it must never enter the session-identity basis.
VOLATILE_BLOCK_PREFIXES = ("x-anthropic-billing-header:",)


def _flatten(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict)
            and not b.get("text", "").startswith(VOLATILE_BLOCK_PREFIXES)
        )
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


MAIN_FINGERPRINT = "You are Claude Code, Anthropic's official CLI"
SUBAGENT_MARKERS = ("Claude Agent SDK", "You are an agent for Claude Code")


def _est_context_tokens(body: dict) -> int:
    """Cheap size estimate for the bounce-or-stick decision; ~4 chars/token."""
    total = 0
    for msg in body.get("messages") or []:
        total += len(str(msg.get("content")))
    return total // 4


def request_role(body: dict) -> str:
    """fast: haiku-class. subagent: Task/SDK agent fingerprints.
    main: the interactive CLI loop, and the safe default for unknowns."""
    if "haiku" in (body.get("model") or ""):
        return "fast"
    system = _flatten(body.get("system") or "")[:KEY_BASIS_CHARS]
    if MAIN_FINGERPRINT in system:
        return "main"
    if any(marker in system for marker in SUBAGENT_MARKERS):
        return "subagent"
    return "main"


def request_capabilities(body: dict) -> set[str]:
    needs: set[str] = set()
    for msg in body.get("messages") or []:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                needs.add("vision")
    return needs


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
            sticky = _est_context_tokens(body) > STICKY_CONTEXT_TOKENS
            if (
                backend
                and not backend.is_down()
                and (sticky or not backend.at_capacity)
                and time.time() - ts < AFFINITY_TTL_S
            ):
                self.affinity[key] = (name, time.time())
                return backend

        role = request_role(body)
        needs = request_capabilities(body)
        if needs:
            capable = [b for b in self.pool.with_capabilities(needs) if not b.at_capacity]
            if capable:
                candidates = capable
            else:
                candidates = [b for b in self.pool.with_role(role) if not b.at_capacity]
        else:
            candidates = [b for b in self.pool.with_role(role) if not b.at_capacity]
        # Hosts whose hardware is already grinding on this role's traffic:
        # overflowing onto a sibling backend there (two servers, one box)
        # adds prefill load to the same GPU instead of adding capacity.
        hot_hosts: set[str] = set()
        if role in ("main", "subagent"):
            # overflow: if every backend for the role is busy (or none up),
            # widen to the other agentic role
            other = "subagent" if role == "main" else "main"
            if not candidates or min(b.in_flight for b in candidates) > 0:
                hot_hosts = {
                    b.host
                    for b in self.pool.with_role(role)
                    if b.in_flight > 0 or b.at_capacity
                }
                candidates = candidates + [
                    b for b in self.pool.with_role(other) if not b.at_capacity
                ]
        if not candidates:
            # everything for the role is down or saturated; degrade to any
            # live backend (capacity becomes soft), then to anything at all
            live = [
                b for b in self.pool.backends
                if not b.is_down() and "candidate" not in b.roles
            ]
            candidates = live or [b for b in self.pool.backends if "candidate" not in b.roles]
            candidates = candidates or self.pool.backends

        # least-loaded; ties go to backends on a cold host, then non-fast
        # backends (fast-role hardware is the cheap tier), then to whoever
        # has served least overall
        chosen = min(
            candidates,
            key=lambda b: (b.in_flight, b.host in hot_hosts, "fast" in b.roles, b.requests),
        )
        # Overflow and degraded placements are temporary: recording them as
        # affinity would pin the session (and, once large, pin it forever via
        # the sticky rule) to a backend that never held its KV prefix. Only a
        # role-correct backend may become the session's home.
        if role in chosen.roles:
            self.affinity[key] = (chosen.name, time.time())
        return chosen
