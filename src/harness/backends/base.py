from __future__ import annotations

from typing import Any, AsyncIterator

import httpx

from harness.config import BackendCfg


class BackendError(Exception):
    """Backend unreachable, errored, or died mid-stream."""


class Backend:
    constrained = False

    def __init__(self, cfg: BackendCfg, client: httpx.AsyncClient) -> None:
        self.cfg = cfg
        self.client = client

    async def stream(self, payload: dict[str, Any]) -> AsyncIterator[dict]:
        raise NotImplementedError
        yield  # pragma: no cover

    def apply_constraint(self, payload: dict[str, Any], schema: dict) -> dict[str, Any]:
        return payload
