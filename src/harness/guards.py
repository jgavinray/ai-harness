"""Deterministic workflow guards over conversation history.

Guards return feedback instead of making policy calls through a model. The
relay owns retry mechanics; this module only decides whether a turn should be
nudged and what the nudge should say.
"""

from __future__ import annotations

from harness.config import Settings
from harness.ir import Conversation, ToolCall, ToolCallPart

EDIT_TOOLS = {"Edit", "MultiEdit"}
VERIFY_WORDS = ("pytest", "test", "check", "npm test", "cargo test", "go test")
DONE_WORDS = ("done", "fixed", "complete", "completed", "implemented", "finished")


def guard_metrics(metrics: dict) -> dict:
    fires = metrics.setdefault("guard_fires", {})
    if not isinstance(fires, dict):
        fires = {}
        metrics["guard_fires"] = fires
    return fires


def increment_guard(metrics: dict, name: str) -> None:
    fires = guard_metrics(metrics)
    fires[name] = fires.get(name, 0) + 1


def _file_arg(call: ToolCall | ToolCallPart) -> str:
    value = call.arguments.get("file_path") or call.arguments.get("path") or ""
    return str(value)


def _read_files(conv: Conversation) -> set[str]:
    out: set[str] = set()
    for turn in conv.turns:
        for part in turn.parts:
            if isinstance(part, ToolCallPart) and part.name == "Read":
                path = _file_arg(part)
                if path:
                    out.add(path)
    return out


def is_verification_command(command: str) -> bool:
    lowered = command.lower()
    return any(word in lowered for word in VERIFY_WORDS)


def has_unverified_edit(conv: Conversation) -> bool:
    edited = False
    for turn in conv.turns:
        for part in turn.parts:
            if not isinstance(part, ToolCallPart):
                continue
            if part.name in EDIT_TOOLS or part.name == "Write":
                edited = True
            elif part.name == "Bash" and is_verification_command(
                str(part.arguments.get("command", ""))
            ):
                edited = False
    return edited


def _done_claim(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in DONE_WORDS)


def guard_tool_call(
    conv: Conversation, call: ToolCall, settings: Settings
) -> tuple[str, str] | None:
    if not settings.pipeline.workflow_guards:
        return None
    path = _file_arg(call)
    if (
        settings.pipeline.guard_edit_without_read
        and call.name in EDIT_TOOLS
        and path
        and path not in _read_files(conv)
    ):
        return (
            "edit_without_read",
            f"Read {path!r} before editing it, then retry the edit with the exact current text.",
        )
    return None


def guard_done_claim(
    conv: Conversation, text: str, settings: Settings
) -> tuple[str, str] | None:
    if not settings.pipeline.workflow_guards or not settings.pipeline.guard_verify_after_edit:
        return None
    if has_unverified_edit(conv) and _done_claim(text):
        return (
            "verify_after_edit",
            "You changed files but have not run a relevant test or check since the edit. "
            "Run a verification command now; only claim completion after it passes.",
        )
    return None
