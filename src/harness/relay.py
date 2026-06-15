"""The relay loop: backend call → validate/repair tool calls → bounded
feedback retries. Yields IR events; the codec turns them into Anthropic SSE.

Invariants:
- a ToolCall is only yielded after validating against the original schema;
- retries never duplicate already-streamed text (suppress_text);
- the loop always terminates with a Done event.
"""

from __future__ import annotations

from dataclasses import replace
from typing import AsyncIterator, Awaitable, Callable

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
from harness.guards import (
    guard_done_claim,
    guard_metrics,
    guard_tool_call,
    has_unverified_edit,
    increment_guard,
    normalize_confused_paths,
)
from harness.skills import SkillCompiler, skill_name
from harness.profiles.base import Profile
from harness.reasoning_budget import apply_reasoning_budget
from harness.repair.degenerate import DegenerateDetector
from harness.repair.toolcalls import repair_toolcall

ReviewCallback = Callable[[str, Conversation, str, dict], Awaitable[str | None]]


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


def _append_guard_feedback(conv: Conversation, guard: str, message: str) -> Conversation:
    turns = conv.turns + (
        Turn("assistant", (TextPart(f"[workflow guard: {guard}]"),)),
        Turn("user", (TextPart(message),)),
    )
    return replace(conv, turns=turns)


def _append_skill_feedback(conv: Conversation, name: str, compiled: str) -> Conversation:
    turns = conv.turns + (
        Turn("assistant", (TextPart(f"[requested skill: {name}]"),)),
        Turn("user", (TextPart(f"Compiled skill procedure for {name}:\n{compiled}"),)),
    )
    return replace(conv, turns=turns)


def _append_invalid_skill_feedback(conv: Conversation, attempted: str) -> Conversation:
    turns = conv.turns + (
        Turn("assistant", (TextPart(f"[invalid Skill call: {attempted[:200]}]"),)),
        Turn("user", (TextPart(
            "That Skill request could not be validated by the harness. Continue "
            "the task directly with concrete tools such as Bash, Read, Grep, "
            "Glob, Edit, or Write; do not wait for a skill."
        ),)),
    )
    return replace(conv, turns=turns)


def _append_tool_required_feedback(conv: Conversation) -> Conversation:
    turns = conv.turns + (
        Turn("assistant", (TextPart("[tool call required after invalid Skill request]"),)),
        Turn("user", (TextPart(
            "Your previous response still did not call a tool. Continue now by "
            "calling Bash, Read, Grep, Glob, Edit, or Write."
        ),)),
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
    reviewer: ReviewCallback | None = None,
    role: str = "main",
    body: dict | None = None,
) -> AsyncIterator[IREvent]:
    m = metrics if metrics is not None else {}
    m.setdefault("retries", 0)
    m.setdefault("repaired_calls", 0)
    m.setdefault("valid_calls", 0)
    m.setdefault("invalid_calls", 0)
    m.setdefault("degenerate_aborts", 0)
    m.setdefault("loop_breaks", 0)
    m.setdefault("tool_surfaced", 0)
    m.setdefault("tool_surfaced_names", [])
    m.setdefault("skill_compiled", 0)
    m.setdefault("plan_drift", 0)
    m.setdefault("path_rewrites", 0)
    m.setdefault("path_rewrite_names", [])
    guard_metrics(m)
    attempts = 0
    suppress_text = False
    constraint_schema: dict | None = None
    skill_compiler = SkillCompiler(settings, profile.name)
    require_tool_after_invalid_skill = False

    model_name = getattr(backend, "model_name", settings.backend.model)

    async def reviewed(trigger: str, message: str) -> str:
        if reviewer is None:
            return message
        feedback = await reviewer(trigger, conv, message, m)
        if not feedback:
            return message
        return f"{message}\n\nReviewer feedback:\n{feedback}"

    while True:
        payload = profile.render(conv, model_name)
        apply_reasoning_budget(payload, settings, backend, role, body or {}, conv, m)
        if attempts and backend.constrained and constraint_schema is not None:
            payload = backend.apply_constraint(payload, constraint_schema)

        detector = DegenerateDetector()
        bad_call: ToolCall | None = None
        bad_error = ""
        loop_call: ToolCall | None = None
        loop_seen = 0
        emitted_valid_call = False
        guarded_call: tuple[str, str] | None = None
        guarded_done: tuple[str, str] | None = None
        skill_feedback: tuple[str, str] | None = None
        invalid_skill: str | None = None
        tool_required_after_invalid_skill = False
        buffered_text: list[str] = []
        buffer_text = (
            settings.pipeline.workflow_guards
            and settings.pipeline.guard_verify_after_edit
            and (
                has_unverified_edit(conv)
                or (settings.planning.enabled and "Plan status: Step" in conv.system)
            )
        )

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
                if isinstance(ev, TextDelta) and buffer_text:
                    buffered_text.append(ev.text)
                    continue
                yield ev
            elif isinstance(ev, ToolCall):
                fixed, error = repair_toolcall(ev, conv.tools)
                if fixed is None:
                    surfaced = _surface_tool(conv, ev.name)
                    if surfaced is not None:
                        conv = surfaced
                        m["tool_surfaced"] += 1
                        m["tool_surfaced_names"].append(ev.name)
                        fixed, error = repair_toolcall(ev, conv.tools)
                if (
                    fixed is None
                    and ev.name == "Skill"
                    and settings.skills.enabled
                    and attempts < settings.pipeline.repair_retries
                ):
                    invalid_skill = ev.raw_arguments or str(ev.arguments)
                    break
                if fixed is not None:
                    fixed, path_rewritten = normalize_confused_paths(fixed)
                    if path_rewritten:
                        m["path_rewrites"] += 1
                        m["path_rewrite_names"].append(fixed.name)
                    if fixed.name == "Skill" and settings.skills.enabled:
                        name = skill_name(fixed.arguments)
                        compiled = skill_compiler.compile(name) if name else None
                        if compiled and attempts < settings.pipeline.repair_retries:
                            skill_feedback = (name, compiled)
                            break
                    guard = guard_tool_call(conv, fixed, settings)
                    if guard is not None and attempts < settings.pipeline.repair_retries:
                        guarded_call = guard
                        break
                    seen = _repeat_count(conv, fixed)
                    if seen >= LOOP_THRESHOLD and attempts < settings.pipeline.repair_retries:
                        loop_call, loop_seen = fixed, seen
                        break
                    emitted_valid_call = True
                    require_tool_after_invalid_skill = False
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
                if buffered_text and not emitted_valid_call and ev.stop_reason != "tool_use":
                    guarded_done = guard_done_claim(conv, "".join(buffered_text), settings)
                    if guarded_done is not None and attempts < settings.pipeline.repair_retries:
                        break
                    for text in buffered_text:
                        yield TextDelta(text)
                if not emitted_valid_call and ev.stop_reason == "tool_use":
                    # every call this turn failed validation and retries are gone
                    yield Done("end_turn", ev.input_tokens, ev.output_tokens)
                else:
                    if (
                        require_tool_after_invalid_skill
                        and not emitted_valid_call
                        and ev.stop_reason != "tool_use"
                        and attempts < settings.pipeline.repair_retries
                    ):
                        tool_required_after_invalid_skill = True
                        break
                    yield ev
                return

        if loop_call is not None:
            attempts += 1
            m["loop_breaks"] += 1
            increment_guard(m, "same_approach")
            suppress_text = True
            feedback = (
                f"You have already called {loop_call.name!r} with these identical "
                f"arguments {loop_seen} times in this conversation; the result "
                "will not change. Do not repeat it. Use the results you already "
                "have, take a different action, or state your conclusion."
            )
            conv = _append_guard_feedback(conv, "same_approach", await reviewed("loop_break", feedback))
            continue

        if guarded_call is not None:
            attempts += 1
            guard, message = guarded_call
            increment_guard(m, guard)
            if guard == "plan_drift":
                m["plan_drift"] += 1
            suppress_text = True
            conv = _append_guard_feedback(conv, guard, await reviewed(guard, message))
            continue

        if skill_feedback is not None:
            attempts += 1
            name, compiled = skill_feedback
            m["skill_compiled"] += 1
            suppress_text = True
            conv = _append_skill_feedback(conv, name, compiled)
            continue

        if invalid_skill is not None:
            attempts += 1
            suppress_text = True
            require_tool_after_invalid_skill = True
            conv = _append_invalid_skill_feedback(conv, invalid_skill)
            continue

        if tool_required_after_invalid_skill:
            attempts += 1
            suppress_text = True
            conv = _append_tool_required_feedback(conv)
            continue

        if guarded_done is not None:
            attempts += 1
            guard, message = guarded_done
            increment_guard(m, guard)
            if guard == "plan_drift":
                m["plan_drift"] += 1
            suppress_text = True
            conv = _append_guard_feedback(conv, guard, await reviewed(guard, message))
            continue

        if bad_call is None:
            # stream ended without a Done (backend quirk); close the turn
            yield Done("tool_use" if emitted_valid_call else "end_turn")
            return

        attempts += 1
        m["retries"] += 1
        suppress_text = True
        tool = next((t for t in conv.tools if t.name == bad_call.name), None)
        constraint_schema = tool.input_schema if tool else None
        if reviewer is not None:
            attempt = bad_call.raw_arguments or str(bad_call.arguments)
            feedback = (
                f"Your call to tool {bad_call.name!r} was invalid: {bad_error}\n"
                f"Your arguments were: {attempt[:500]}\n"
                "Call the tool again with corrected JSON arguments that match its schema exactly."
            )
            reviewed_feedback = await reviewed("invalid_tool_retry", feedback)
            conv = replace(
                conv,
                turns=conv.turns + (
                    Turn("assistant", (TextPart(
                        f"[attempted tool call: {bad_call.name} {attempt[:200]}]"
                    ),)),
                    Turn("user", (TextPart(reviewed_feedback),)),
                ),
            )
        else:
            conv = _append_feedback(conv, bad_call, bad_error)
