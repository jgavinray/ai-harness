"""Exact-match response cache: identical rendered payloads skip inference.

Targets Claude Code's repetitive haiku-class background calls and automatic
retries. Keyed on the full rendered backend payload (minus stream flags),
so a hit is by construction the same question to the same model.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict

from harness.ir import IREvent

EXCLUDED_KEYS = ("stream", "stream_options")


def payload_key(payload: dict) -> str:
    basis = {k: v for k, v in payload.items() if k not in EXCLUDED_KEYS}
    return hashlib.sha256(
        json.dumps(basis, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class ResponseCache:
    def __init__(self, ttl_s: float = 600.0, max_entries: int = 256) -> None:
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self.store: OrderedDict[str, tuple[float, list[IREvent]]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> list[IREvent] | None:
        entry = self.store.get(key)
        if entry is None or time.time() - entry[0] > self.ttl_s:
            self.store.pop(key, None)
            self.misses += 1
            return None
        self.store.move_to_end(key)
        self.hits += 1
        return entry[1]

    def put(self, key: str, events: list[IREvent]) -> None:
        self.store[key] = (time.time(), events)
        self.store.move_to_end(key)
        while len(self.store) > self.max_entries:
            self.store.popitem(last=False)
