"""Detect runaway repetition in streamed text.

Uses the minimal-period property: a string t has period p iff t occurs in
t+t at offset p. If the last TAIL chars are fully periodic with at least
MIN_CYCLES repetitions, the stream is degenerating.
"""

from __future__ import annotations

WINDOW = 4000
CHECK_EVERY = 64
TAIL = 240
MIN_CYCLES = 3


class DegenerateDetector:
    def __init__(self) -> None:
        self.tail = ""
        self.since_check = 0

    def feed(self, text: str) -> bool:
        self.tail = (self.tail + text)[-WINDOW:]
        self.since_check += len(text)
        if self.since_check < CHECK_EVERY or len(self.tail) < TAIL:
            return False
        self.since_check = 0
        t = self.tail[-TAIL:]
        period = (t + t).find(t, 1)
        return 0 < period <= TAIL // MIN_CYCLES
