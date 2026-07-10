"""Lightweight runtime action-state detection.

This is deliberately local protocol shaping, not task planning. The state only
describes which tool surface is mechanically legal for the next backend call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from harness.config import Settings
from harness.guards import (
    SHORT_INSTRUCTION_MAX_CHARS,
    is_verify_instruction,
    unverified_edit_count,
)
from harness.ir import Conversation, TextPart, ToolCallPart

READONLY_TOOLS = ("Read", "Grep", "Glob", "LS", "WebFetch", "WebSearch")
# Write stays available outside verify state: creating a new file is
# legitimate work in any phase and has no read-first precondition.
INSPECT_TOOLS = READONLY_TOOLS + ("Bash", "Write")
EDIT_TOOLS = ("Edit", "MultiEdit")
CREATE_TOOLS = ("Write", "Bash")
VERIFY_TOOLS = ("Bash",) + READONLY_TOOLS
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


def _has_inspect_intent(text: str) -> bool:
    return bool(re.search(r"\b(read|inspect|review|search|find|open|list)\b|look at", text))


def current_action_state(conv: Conversation, settings: Settings) -> ActionState:
    if not settings.pipeline.action_state_tools:
        return ActionState("unrestricted", ())

    # Bind verify only after a whole change-set of unverified edits: renames
    # and refactors need consecutive edits, and binding at the first edit made
    # them impossible to finish. The done-claim gate still demands
    # verification for any unverified edit. The plan status line must never
    # bind state: its step position tracks tool-call count, not real
    # progress, so long sessions pin to the final "Verify ..." step and a
    # plan-keyed binding locks them there (live lockout, 2026-07-09).
    if unverified_edit_count(conv) >= settings.pipeline.unverified_edit_limit:
        return ActionState(
            "verify",
            VERIFY_TOOLS,
            requires_tool=True,
            required_tool="Bash",
            reason="unverified_edit",
        )

    latest = _latest_user_text(conv)
    # Free-text intent only binds for short imperative instructions ("run
    # tests"); long task briefs that mention verify/create as steps must not
    # lock the session's tool surface (eval brick1-verify: multi-step 0/5).
    if is_verify_instruction(latest):
        return ActionState("verify", VERIFY_TOOLS, requires_tool=True, required_tool="Bash", reason="verify_request")
    if len(latest.strip()) <= SHORT_INSTRUCTION_MAX_CHARS and any(word in latest for word in CREATE_WORDS):
        return ActionState("create_file", CREATE_TOOLS, requires_tool=True, reason="create_request")
    if not settings.pipeline.guard_edit_without_read:
        return ActionState("edit_existing", EDIT_TOOLS + ("Write", "Bash") + READONLY_TOOLS, reason="edit_guard_relaxed")
    if _read_seen(conv):
        return ActionState("edit_existing", EDIT_TOOLS + ("Write", "Bash") + READONLY_TOOLS, reason="file_read")
    requires_tool = (
        len(latest.strip()) <= SHORT_INSTRUCTION_MAX_CHARS
        and _has_inspect_intent(latest)
    )
    return ActionState("inspect", INSPECT_TOOLS, requires_tool=requires_tool, reason="no_file_read")


def shape_tools_for_state(conv: Conversation, state: ActionState) -> Conversation:
    if not state.allowed_tools:
        return conv
    allowed = set(state.allowed_tools)
    shaped = tuple(tool for tool in conv.tools if tool.name in allowed)
    if not shaped:
        return conv
    return replace(conv, tools=shaped)
