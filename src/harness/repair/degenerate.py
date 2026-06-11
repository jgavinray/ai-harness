"""Detect runaway repetition in streamed text.

Checks (cheaply, every CHECK_EVERY chars) whether the tail of the stream is
the same block repeated three times at several block lengths.
"""

from __future__ import annotations

WINDOW = 4000
CHECK_EVERY = 64
BLOCK_LENGTHS = (24, 48, 96, 192)


class DegenerateDetector:
    def __init__(self) -> None:
        self.tail = ""
        self.since_check = 0

    def feed(self, text: str) -> bool:
        self.tail = (self.tail + text)[-WINDOW:]
        self.since_check += len(text)
        if self.since_check < CHECK_EVERY:
            return False
        self.since_check = 0
        for length in BLOCK_LENGTHS:
            if len(self.tail) >= 3 * length:
                a = self.tail[-length:]
                if a == self.tail[-2 * length : -length] == self.tail[-3 * length : -2 * length]:
                    return True
        return False
