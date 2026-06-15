"""Lightweight runtime action-state detection.

This is deliberately local protocol shaping, not task planning. The state only
describes which tool surface is mechanically legal for the next backend call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from harness.config import Settings
from harness.guards import has_unverified_edit
from harness.ir import Conversation, TextPart, ToolCallPart
from harness.planning import plan_status

INSPECT_TOOLS = ("Read", "Grep", "Glob", "LS")
EDIT_TOOLS = ("Edit", "MultiEdit")
CREATE_TOOLS = ("Write", "Bash")
VERIFY_TOOLS = ("Bash",) + INSPECT_TOOLS
CREATE_WORDS = ("create", "new file", "add file", "write a file")
VERIFY_WORDS = ("verify", "check", "run tests", "build", "compile", "lint")


@dataclass(frozen=True)
class ActionState:
    name: str
    allowed_tools: tuple[str, ...]
    requires_tool: bool = False
    required_tool: str | None = None
    reason: str | None = None


def _latest_user_text(conv: Conversation) -> str:
    for turn in reversed(conv.turns):
        if turn.role != "user":
            continue
        texts = [p.text for p in turn.parts if isinstance(p, TextPart)]
        if texts:
            return "\n".join(texts).lower()
    return ""


def _read_seen(conv: Conversation) -> bool:
    return any(
        isinstance(part, ToolCallPart) and part.name == "Read"
        for turn in conv.turns
        for part in turn.parts
    )


def _is_verify_step(step: str) -> bool:
    return _has_verify_intent(step.lower())


def _has_verify_intent(text: str) -> bool:
    return bool(
        re.search(
            r"\b(verify|check|build|compile|lint)\b|\brun(?:ning)?\s+(?:the\s+)?tests?\b",
            text,
        )
    )


def _has_inspect_intent(text: str) -> bool:
    return bool(re.search(r"\b(read|inspect|review|search|find|open|list)\b|look at", text))


def current_action_state(conv: Conversation, settings: Settings) -> ActionState:
    if not settings.pipeline.action_state_tools:
        return ActionState("unrestricted", ())

    plan = plan_status(conv.system)
    if has_unverified_edit(conv):
        return ActionState(
            "verify",
            VERIFY_TOOLS,
            requires_tool=True,
            required_tool="Bash",
            reason="unverified_edit",
        )
    if settings.planning.enabled and plan is not None and _is_verify_step(plan[2]):
        return ActionState(
            "verify",
            VERIFY_TOOLS,
            requires_tool=True,
            required_tool="Bash",
            reason="plan_verify_step",
        )

    latest = _latest_user_text(conv)
    if _has_verify_intent(latest):
        return ActionState("verify", VERIFY_TOOLS, requires_tool=True, required_tool="Bash", reason="verify_request")
    if any(word in latest for word in CREATE_WORDS):
        return ActionState("create_file", CREATE_TOOLS, requires_tool=True, reason="create_request")
    if _read_seen(conv):
        return ActionState("edit_existing", EDIT_TOOLS + INSPECT_TOOLS, reason="file_read")
    return ActionState("inspect", INSPECT_TOOLS, requires_tool=_has_inspect_intent(latest), reason="no_file_read")


def shape_tools_for_state(conv: Conversation, state: ActionState) -> Conversation:
    if not state.allowed_tools:
        return conv
    allowed = set(state.allowed_tools)
    shaped = tuple(tool for tool in conv.tools if tool.name in allowed)
    if not shaped:
        return conv
    return replace(conv, tools=shaped)
