"""Neutral intermediate representation of a conversation and of model output.

Codecs (Anthropic side) and profiles (backend side) are the only modules that
touch wire formats; everything in between operates on these types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Union


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    input_schema: dict[str, Any]  # what the model sees (possibly simplified)
    original_schema: dict[str, Any]  # what Claude Code sent; validation target


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ThinkingPart:
    text: str


@dataclass(frozen=True)
class ToolCallPart:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResultPart:
    tool_call_id: str
    content: str
    is_error: bool = False


Part = Union[TextPart, ThinkingPart, ToolCallPart, ToolResultPart]


@dataclass(frozen=True)
class Turn:
    role: Literal["user", "assistant"]
    parts: tuple[Part, ...]


@dataclass(frozen=True)
class GenParams:
    max_tokens: int
    temperature: float | None = None
    stop_sequences: tuple[str, ...] = ()
    stream: bool = False


@dataclass(frozen=True)
class Conversation:
    system: str
    turns: tuple[Turn, ...]
    tools: tuple[ToolDef, ...]
    params: GenParams


# ---- stream events (profile parse / relay output) ----


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ThinkingDelta:
    text: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""  # original string when JSON parse failed


@dataclass(frozen=True)
class Done:
    stop_reason: str  # end_turn | tool_use | max_tokens
    input_tokens: int = 0
    output_tokens: int = 0


IREvent = Union[TextDelta, ThinkingDelta, ToolCall, Done]
