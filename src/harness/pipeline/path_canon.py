"""Normalize known-bad path aliases before prompts reach backend models."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from harness.config import Settings
from harness.guards import BAD_DEV_PR_PREFIX, GOOD_DEV_PR_PREFIX
from harness.ir import Conversation, TextPart, ThinkingPart, ToolCallPart, ToolResultPart, Turn

ALIASES = ((BAD_DEV_PR_PREFIX, GOOD_DEV_PR_PREFIX),)


def canonicalize_text(text: str) -> str:
    out = text
    for bad, good in ALIASES:
        out = out.replace(bad, good)
    return out


def _canon_value(value: Any) -> Any:
    if isinstance(value, str):
        return canonicalize_text(value)
    if isinstance(value, list):
        return [_canon_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _canon_value(item) for key, item in value.items()}
    return value


def _canon_turn(turn: Turn) -> Turn:
    parts = []
    changed = False
    for part in turn.parts:
        if isinstance(part, TextPart):
            text = canonicalize_text(part.text)
            changed = changed or text != part.text
            parts.append(replace(part, text=text) if text != part.text else part)
        elif isinstance(part, ThinkingPart):
            text = canonicalize_text(part.text)
            changed = changed or text != part.text
            parts.append(replace(part, text=text) if text != part.text else part)
        elif isinstance(part, ToolResultPart):
            content = canonicalize_text(part.content)
            changed = changed or content != part.content
            parts.append(replace(part, content=content) if content != part.content else part)
        elif isinstance(part, ToolCallPart):
            args = _canon_value(part.arguments)
            changed = changed or args != part.arguments
            parts.append(replace(part, arguments=args) if args != part.arguments else part)
        else:
            parts.append(part)
    return Turn(turn.role, tuple(parts)) if changed else turn


class PathCanonStage:
    def apply(
        self, conv: Conversation, settings: Settings, metrics: dict | None = None
    ) -> Conversation:
        system = canonicalize_text(conv.system)
        turns = tuple(_canon_turn(turn) for turn in conv.turns)
        changed = system != conv.system or turns != conv.turns
        if metrics is not None:
            metrics["path_canonicalized"] = changed
        if not changed:
            return conv
        return replace(conv, system=system, turns=turns)
