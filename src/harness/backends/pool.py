"""Backend fleet: per-backend state, circuit breaking, stream wrapping.

Single-backend configs become a one-entry pool so the server has exactly
one code path.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator
from urllib.parse import urlparse

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
        self.output_tokens = 0
        self.ttft_ms: list[int] = []
        self.recent_cache: list[tuple[int, int]] = []  # (prompt, cached) per request
        # session -> tokens of its last request here, insertion-ordered by
        # recency; basis for the llama.cpp KV-residency estimate
        self.kv_resident: dict[str, int] = {}
        # last good KV-occupancy reading, held across missed polls
        self.kv_used: dict | None = None
        self.kv_used_ts = 0.0
        # last vLLM Prometheus token-counter sample for live rates
        self.live_token_counters: dict[str, float] = {}
        self.live_token_ts = 0.0

    @property
    def model_name(self) -> str:
        return self.cfg.model

    @property
    def host(self) -> str:
        """Hostname only (port stripped): two servers on one box share a host."""
        return urlparse(self.cfg.base_url).hostname or self.cfg.base_url

    @property
    def constrained(self) -> bool:
        return self.backend.constrained

    def apply_constraint(self, payload: dict[str, Any], schema: dict) -> dict[str, Any]:
        return self.backend.apply_constraint(payload, schema)

    async def stream(self, payload: dict[str, Any]) -> AsyncIterator[dict]:
        self.in_flight += 1
        self.requests += 1
        try:
            async for chunk in self.backend.stream(payload):
                yield chunk
            self.mark_ok()
        except BackendError:
            self.trip()
            raise
        finally:
            self.in_flight -= 1

    def is_down(self) -> bool:
        return time.time() < self.cooldown_until

    @property
    def at_capacity(self) -> bool:
        return (
            self.cfg.max_in_flight is not None
            and self.in_flight >= self.cfg.max_in_flight
        )

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
            capabilities=[],
        )
    ]


class BackendPool:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.client = client
        self.backends = [PooledBackend(cfg, client) for cfg in _fleet_from(settings)]

    def reconfigure(self, settings: Settings) -> dict:
        """Diff the new [[backends]] list against the live pool by name.

        Surviving backends are updated in place so their counters, breaker
        state, and any in-flight requests carry over; in-flight streams on
        removed backends hold their own reference and finish normally.
        """
        by_name = {b.name: b for b in self.backends}
        summary: dict = {"updated": [], "added": [], "removed": []}
        rebuilt = []
        for cfg in _fleet_from(settings):
            b = by_name.pop(cfg.name, None)
            if b:
                b.cfg = cfg
                b.roles = list(cfg.roles)
                b.backend = make_backend(cfg, self.client)
                b.profile = get_profile(cfg.profile)
                summary["updated"].append(cfg.name)
            else:
                b = PooledBackend(cfg, self.client)
                summary["added"].append(cfg.name)
            rebuilt.append(b)
        summary["removed"] = sorted(by_name)
        self.backends = rebuilt
        return summary

    def get(self, name: str) -> PooledBackend | None:
        return next((b for b in self.backends if b.name == name), None)

    def with_role(self, role: str, include_down: bool = False) -> list[PooledBackend]:
        return [
            b for b in self.backends
            if role in b.roles
            and "candidate" not in b.roles
            and (include_down or not b.is_down())
        ]

    def with_capabilities(self, needs: set[str]) -> list[PooledBackend]:
        return [
            b for b in self.backends
            if needs.issubset(set(b.cfg.capabilities))
            and "candidate" not in b.roles
            and not b.is_down()
        ]

    async def stream(self, b: PooledBackend, payload: dict[str, Any]) -> AsyncIterator[dict]:
        async for chunk in b.stream(payload):
            yield chunk
