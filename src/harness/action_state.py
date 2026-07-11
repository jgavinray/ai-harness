"""Lightweight runtime action-state detection.

This is deliberately local protocol shaping, not task planning. The state only
describes which tool surface is mechanically legal for the next backend call.

Invariant (spec 2026-07-11-default-open-enforcement): states are SUBTRACTIVE.
A state may only block tools whose misuse is named by a measured failure, and
must never enumerate the permitted surface — enumerated allowlists silently
deny every tool the author didn't foresee, including all future client tools
(live regression 2026-07-11: `Agent` was absent from every allowlist, so
subagent fan-out was structurally impossible and the user's explicit "fan out
subagents" was denied 9+ times). Unknown tools always pass.
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

# The one evidence-backed subtraction: verify pressure blocks further edits
# until a check runs (unverified-edit pileups, consistency spec 2026-07-07).
# NotebookEdit is an edit tool by any other name.
EDIT_TOOLS = ("Edit", "MultiEdit")
VERIFY_BLOCKED = EDIT_TOOLS + ("Write", "NotebookEdit")
CREATE_WORDS = ("create", "new file", "add file", "write a file")
VERIFY_WORDS = ("verify", "check", "run tests", "build", "compile", "lint")


@dataclass(frozen=True)
class ActionState:
    name: str
    blocked_tools: tuple[str, ...] = ()
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
        return ActionState("unrestricted")

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
            VERIFY_BLOCKED,
            requires_tool=True,
            required_tool="Bash",
            reason="unverified_edit",
        )

    latest = _latest_user_text(conv)
    # Free-text intent only binds for short imperative instructions ("run
    # tests"); long task briefs that mention verify/create as steps must not
    # lock the session's tool surface (eval brick1-verify: multi-step 0/5).
    if is_verify_instruction(latest):
        return ActionState(
            "verify",
            VERIFY_BLOCKED,
            requires_tool=True,
            required_tool="Bash",
            reason="verify_request",
        )
    if len(latest.strip()) <= SHORT_INSTRUCTION_MAX_CHARS and any(word in latest for word in CREATE_WORDS):
        return ActionState("create_file", requires_tool=True, reason="create_request")
    # inspect vs edit_existing is telemetry, not enforcement: read-before-edit
    # is guard_edit_without_read's job (per-file, with actionable feedback);
    # the old state-level Edit hiding was a coarser duplicate of that guard,
    # and its inspect→edit_existing catalog flip broke prefix stability
    # (Law 2) by changing the rendered tool list mid-session.
    if not settings.pipeline.guard_edit_without_read:
        return ActionState("edit_existing", reason="edit_guard_relaxed")
    if _read_seen(conv):
        return ActionState("edit_existing", reason="file_read")
    requires_tool = (
        len(latest.strip()) <= SHORT_INSTRUCTION_MAX_CHARS
        and _has_inspect_intent(latest)
    )
    return ActionState("inspect", requires_tool=requires_tool, reason="no_file_read")


def shape_tools_for_state(conv: Conversation, state: ActionState) -> Conversation:
    if not state.blocked_tools:
        return conv
    shaped = tuple(tool for tool in conv.tools if tool.name not in state.blocked_tools)
    if not shaped:
        return conv
    return replace(conv, tools=shaped)
