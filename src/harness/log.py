"""Per-request JSONL logging — the single instrumentation source shared by
production observability and the eval suite."""

from __future__ import annotations

import json
import threading
from pathlib import Path


class RequestLogger:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        if not self.path:
            return
        line = json.dumps(record, separators=(",", ":"))
        with self._lock, self.path.open("a") as f:
            f.write(line + "\n")
