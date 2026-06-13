"""The relay loop: backend call → validate/repair tool calls → bounded
feedback retries. Yields IR events; the codec turns them into Anthropic SSE.

Invariants:
- a ToolCall is only yielded after validating against the original schema;
- retries never duplicate already-streamed text (suppress_text);
- the loop always terminates with a Done event.
"""

from __future__ import annotations

from dataclasses import replace
from typing import AsyncIterator

from harness.backends.base import Backend
from harness.config import Settings
from harness.ir import (
    Conversation,
    Done,
    IREvent,
    TextDelta,
    TextPart,
    ThinkingDelta,
    ToolCall,
    ToolCallPart,
    Turn,
)
from harness.profiles.base import Profile
from harness.repair.degenerate import DegenerateDetector
from harness.repair.toolcalls import repair_toolcall


# Cross-turn loop breaking: the DegenerateDetector catches repetition inside
# one response; nothing else stops a model re-running the identical command
# turn after turn (observed: a 400-turn `git worktree list` loop).
LOOP_THRESHOLD = 3
LOOP_WINDOW_TURNS = 12


def _repeat_count(conv: Conversation, call: ToolCall) -> int:
    n = 0
    for turn in conv.turns[-LOOP_WINDOW_TURNS:]:
        if turn.role != "assistant":
            continue
        for p in turn.parts:
            if (
                isinstance(p, ToolCallPart)
                and p.name == call.name
                and p.arguments == call.arguments
            ):
                n += 1
    return n


def _append_loop_feedback(conv: Conversation, call: ToolCall, seen: int) -> Conversation:
    feedback = (
        f"You have already called {call.name!r} with these identical arguments "
        f"{seen} times in this conversation; the result will not change. "
        "Do not repeat it. Use the results you already have, take a different "
        "action, or state your conclusion."
    )
    turns = conv.turns + (
        Turn("assistant", (TextPart(f"[repeated tool call suppressed: {call.name}]"),)),
        Turn("user", (TextPart(feedback),)),
    )
    return replace(conv, turns=turns)


def _append_feedback(conv: Conversation, bad: ToolCall, error: str) -> Conversation:
    attempt = bad.raw_arguments or str(bad.arguments)
    feedback = (
        f"Your call to tool {bad.name!r} was invalid: {error}\n"
        f"Your arguments were: {attempt[:500]}\n"
        f"Call the tool again with corrected JSON arguments that match its schema exactly."
    )
    turns = conv.turns + (
        Turn("assistant", (TextPart(f"[attempted tool call: {bad.name} {attempt[:200]}]"),)),
        Turn("user", (TextPart(feedback),)),
    )
    return replace(conv, turns=turns)


def _surface_tool(conv: Conversation, name: str) -> Conversation | None:
    """The model called a catalogued tool whose schema is not surfaced.
    Returns conv with the real ToolDef added (so validation, feedback,
    and constrained retries all see it), or None if the name is unknown."""
    if any(t.name == name for t in conv.tools):
        return None
    tool = next((t for t in conv.all_tools if t.name == name), None)
    if tool is None:
        return None
    return replace(conv, tools=conv.tools + (tool,))


async def run(
    conv: Conversation,
    profile: Profile,
    backend: Backend,
    settings: Settings,
    metrics: dict | None = None,
) -> AsyncIterator[IREvent]:
    m = metrics if metrics is not None else {}
    m.setdefault("retries", 0)
    m.setdefault("repaired_calls", 0)
    m.setdefault("valid_calls", 0)
    m.setdefault("invalid_calls", 0)
    m.setdefault("degenerate_aborts", 0)
    m.setdefault("loop_breaks", 0)
    m.setdefault("tool_surfaced", 0)
    attempts = 0
    suppress_text = False
    constraint_schema: dict | None = None

    model_name = getattr(backend, "model_name", settings.backend.model)
    while True:
        payload = profile.render(conv, model_name)
        if attempts and backend.constrained and constraint_schema is not None:
            payload = backend.apply_constraint(payload, constraint_schema)

        detector = DegenerateDetector()
        bad_call: ToolCall | None = None
        bad_error = ""
        loop_call: ToolCall | None = None
        loop_seen = 0
        emitted_valid_call = False

        async for ev in profile.parse(backend.stream(payload)):
            if isinstance(ev, (TextDelta, ThinkingDelta)):
                if suppress_text:
                    continue
                if isinstance(ev, TextDelta) and detector.feed(ev.text):
                    m["degenerate_aborts"] += 1
                    yield TextDelta("\n[output truncated: repetition detected]")
                    yield Done("end_turn")
                    return
                if isinstance(ev, ThinkingDelta) and settings.pipeline.reasoning == "strip":
                    continue
                yield ev
            elif isinstance(ev, ToolCall):
                fixed, error = repair_toolcall(ev, conv.tools)
                if fixed is None:
                    surfaced = _surface_tool(conv, ev.name)
                    if surfaced is not None:
                        conv = surfaced
                        m["tool_surfaced"] += 1
                        fixed, error = repair_toolcall(ev, conv.tools)
                if fixed is not None:
                    seen = _repeat_count(conv, fixed)
                    if seen >= LOOP_THRESHOLD and attempts < settings.pipeline.repair_retries:
                        loop_call, loop_seen = fixed, seen
                        break
                    emitted_valid_call = True
                    m["valid_calls"] += 1
                    if ev.raw_arguments:  # arrived malformed, json-repaired locally
                        m["repaired_calls"] += 1
                    yield fixed
                elif attempts < settings.pipeline.repair_retries:
                    bad_call, bad_error = ev, error or "invalid"
                    break
                else:
                    m["invalid_calls"] += 1
                    raw = ev.raw_arguments or str(ev.arguments)
                    yield TextDelta(f"\n[invalid tool call {ev.name}: {bad_error or error}]\n{raw[:500]}")
            else:  # Done
                if not emitted_valid_call and ev.stop_reason == "tool_use":
                    # every call this turn failed validation and retries are gone
                    yield Done("end_turn", ev.input_tokens, ev.output_tokens)
                else:
                    yield ev
                return

        if loop_call is not None:
            attempts += 1
            m["loop_breaks"] += 1
            suppress_text = True
            conv = _append_loop_feedback(conv, loop_call, loop_seen)
            continue

        if bad_call is None:
            # stream ended without a Done (backend quirk); close the turn
            yield Done("tool_use" if emitted_valid_call else "end_turn")
            return

        attempts += 1
        m["retries"] += 1
        suppress_text = True
        tool = next((t for t in conv.tools if t.name == bad_call.name), None)
        constraint_schema = tool.original_schema if tool else None
        conv = _append_feedback(conv, bad_call, bad_error)
