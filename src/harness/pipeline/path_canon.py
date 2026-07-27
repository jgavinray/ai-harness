"""Normalize configured known-bad path aliases before prompts reach backend models."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from harness.config import Settings
from harness.ir import (
    Conversation,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
    Turn,
)

Aliases = tuple[tuple[str, str], ...]


def canonicalize_text(text: str, aliases: Aliases) -> str:
    out = text
    for bad, good in aliases:
        out = out.replace(bad, good)
    return out


def _canon_value(value: Any, aliases: Aliases) -> Any:
    if isinstance(value, str):
        return canonicalize_text(value, aliases)
    if isinstance(value, list):
        return [_canon_value(item, aliases) for item in value]
    if isinstance(value, dict):
        return {key: _canon_value(item, aliases) for key, item in value.items()}
    return value


def _canon_turn(turn: Turn, aliases: Aliases) -> Turn:
    parts = []
    changed = False
    for part in turn.parts:
        if isinstance(part, (TextPart, ThinkingPart)):
            text = canonicalize_text(part.text, aliases)
            changed = changed or text != part.text
            parts.append(replace(part, text=text) if text != part.text else part)
        elif isinstance(part, ToolResultPart):
            content = canonicalize_text(part.content, aliases)
            changed = changed or content != part.content
            parts.append(replace(part, content=content) if content != part.content else part)
        elif isinstance(part, ToolCallPart):
            args = _canon_value(part.arguments, aliases)
            changed = changed or args != part.arguments
            parts.append(replace(part, arguments=args) if args != part.arguments else part)
        else:
            parts.append(part)
    return Turn(turn.role, tuple(parts)) if changed else turn


class PathCanonStage:
    def apply(
        self, conv: Conversation, settings: Settings, metrics: dict | None = None
    ) -> Conversation:
        aliases: Aliases = tuple(
            (bad, good) for bad, good in settings.pipeline.path_aliases
        )
        if not aliases:
            if metrics is not None:
                metrics["path_canonicalized"] = False
            return conv
        system = canonicalize_text(conv.system, aliases)
        turns = tuple(_canon_turn(turn, aliases) for turn in conv.turns)
        changed = system != conv.system or turns != conv.turns
        if metrics is not None:
            metrics["path_canonicalized"] = changed
        if not changed:
            return conv
        return replace(conv, system=system, turns=turns)
