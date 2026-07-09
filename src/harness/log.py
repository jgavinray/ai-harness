"""Per-request JSONL logging — the single instrumentation source shared by
production observability and the eval suite.

Two modes: a single file (eval runner, legacy) or a directory of daily
partitions (`<dir>/YYYY-MM-DD.jsonl`) — the data-plane layout the flywheel
jobs and the DuckDB index consume.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from harness.rotation import DEFAULT_ROTATE_BYTES, rotate_if_needed


class RequestLogger:
    def __init__(
        self,
        path: str | Path | None,
        rotate_bytes: int = DEFAULT_ROTATE_BYTES,
        directory: str | Path | None = None,
    ) -> None:
        self.directory = Path(directory) if directory else None
        self.path = None if self.directory else (Path(path) if path else None)
        self.rotate_bytes = rotate_bytes
        self._lock = threading.Lock()
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
        elif self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            rotate_if_needed(self.path, self.rotate_bytes)

    def _target(self) -> Path | None:
        if self.directory:
            return self.directory / f"{time.strftime('%Y-%m-%d')}.jsonl"
        return self.path

    def write(self, record: dict) -> None:
        target = self._target()
        if target is None:
            return
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            rotate_if_needed(target, self.rotate_bytes)
            with target.open("a") as f:
                f.write(line + "\n")
