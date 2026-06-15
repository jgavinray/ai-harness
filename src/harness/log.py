"""Per-request JSONL logging — the single instrumentation source shared by
production observability and the eval suite."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from harness.rotation import DEFAULT_ROTATE_BYTES, rotate_if_needed


class RequestLogger:
    def __init__(
        self,
        path: str | Path | None,
        rotate_bytes: int = DEFAULT_ROTATE_BYTES,
    ) -> None:
        self.path = Path(path) if path else None
        self.rotate_bytes = rotate_bytes
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            rotate_if_needed(self.path, self.rotate_bytes)

    def write(self, record: dict) -> None:
        if not self.path:
            return
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            rotate_if_needed(self.path, self.rotate_bytes)
            with self.path.open("a") as f:
                f.write(line + "\n")
