"""Token counting for budgets and the count_tokens endpoint.

Heuristic by default so tests and offline use need no tokenizer downloads;
the TokenCounter protocol is the seam for a real HF tokenizer later.
"""

from __future__ import annotations

import json
from typing import Protocol

from harness.ir import (
    Conversation,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
)

PER_MESSAGE_OVERHEAD = 8


class TokenCounter(Protocol):
    def count_text(self, text: str) -> int: ...


class HeuristicCounter:
    """~chars/3.6, floored at word count — close enough for budgeting."""

    def count_text(self, text: str) -> int:
        return max(round(len(text) / 3.6), len(text.split()))


def _part_text(part) -> str:
    if isinstance(part, (TextPart, ThinkingPart)):
        return part.text
    if isinstance(part, ToolCallPart):
        return part.name + json.dumps(part.arguments)
    if isinstance(part, ToolResultPart):
        return part.content
    return ""


def count_conversation(conv: Conversation, counter: TokenCounter) -> int:
    total = counter.count_text(conv.system)
    for tool in conv.tools:
        total += counter.count_text(tool.description + json.dumps(tool.input_schema))
    for turn in conv.turns:
        total += PER_MESSAGE_OVERHEAD
        for part in turn.parts:
            total += counter.count_text(_part_text(part))
    return total
