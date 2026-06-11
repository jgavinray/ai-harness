"""Backend fleet: per-backend state, circuit breaking, stream wrapping.

Single-backend configs become a one-entry pool so the server has exactly
one code path.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

import httpx

from harness.backends.base import BackendError
from harness.backends.openai_compat import make_backend
from harness.config import PoolBackendCfg, Settings
from harness.profiles.base import Profile
from harness.profiles.registry import get_profile

COOLDOWN_S = 30.0


class PooledBackend:
    def __init__(self, cfg: PoolBackendCfg, client: httpx.AsyncClient) -> None:
        self.name = cfg.name
        self.cfg = cfg
        self.roles = list(cfg.roles)
        self.backend = make_backend(cfg, client)
        self.profile: Profile = get_profile(cfg.profile)
        self.in_flight = 0
        self.consecutive_errors = 0
        self.cooldown_until = 0.0
        # rolling counters for /stats
        self.requests = 0
        self.errors = 0
        self.cached_tokens = 0
        self.prompt_tokens = 0
        self.ttft_ms: list[int] = []

    def is_down(self) -> bool:
        return time.time() < self.cooldown_until

    def trip(self, cooldown_s: float = COOLDOWN_S) -> None:
        self.consecutive_errors += 1
        self.errors += 1
        self.cooldown_until = time.time() + cooldown_s

    def mark_ok(self) -> None:
        self.consecutive_errors = 0
        self.cooldown_until = 0.0


def _fleet_from(settings: Settings) -> list[PoolBackendCfg]:
    if settings.backends:
        return settings.backends
    return [
        PoolBackendCfg(
            name="default",
            kind=settings.backend.kind,
            base_url=settings.backend.base_url,
            model=settings.backend.model,
            api_key=settings.backend.api_key,
            profile=settings.profile.name,
            context_window=settings.profile.context_window,
            roles=["main", "subagent", "fast"],
        )
    ]


class BackendPool:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.backends = [PooledBackend(cfg, client) for cfg in _fleet_from(settings)]

    def get(self, name: str) -> PooledBackend | None:
        return next((b for b in self.backends if b.name == name), None)

    def with_role(self, role: str, include_down: bool = False) -> list[PooledBackend]:
        return [
            b for b in self.backends
            if role in b.roles and (include_down or not b.is_down())
        ]

    async def stream(self, b: PooledBackend, payload: dict[str, Any]) -> AsyncIterator[dict]:
        b.in_flight += 1
        b.requests += 1
        try:
            async for chunk in b.backend.stream(payload):
                yield chunk
            b.mark_ok()
        except BackendError:
            b.trip()
            raise
        finally:
            b.in_flight -= 1
