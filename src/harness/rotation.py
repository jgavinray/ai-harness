"""Size-based rotation for harness JSONL files."""

from __future__ import annotations

import time
from pathlib import Path

DEFAULT_ROTATE_BYTES = 100 * 1024 * 1024


def rotated_path(path: Path, epoch: int | None = None) -> Path:
    ts = int(time.time()) if epoch is None else epoch
    date = time.strftime("%Y%m%d", time.localtime(ts))
    suffix = path.suffix
    stem = path.name[: -len(suffix)] if suffix else path.name
    candidate = path.with_name(f"{stem}-{date}-{ts}{suffix}")
    index = 1
    while candidate.exists():
        candidate = path.with_name(f"{stem}-{date}-{ts}-{index}{suffix}")
        index += 1
    return candidate


def rotate_if_needed(
    path: Path, max_bytes: int = DEFAULT_ROTATE_BYTES
) -> Path | None:
    if max_bytes <= 0:
        return None
    try:
        if not path.exists() or path.stat().st_size < max_bytes:
            return None
        target = rotated_path(path)
        path.rename(target)
        return target
    except FileNotFoundError:
        return None
