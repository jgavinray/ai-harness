"""The relay loop: backend call → validate/repair tool calls → bounded
feedback retries. Yields IR events; the codec turns them into Anthropic SSE.

Invariants:
- a ToolCall is only yielded after validating against the original schema;
- retries never duplicate already-streamed text (suppress_text);
- the loop always terminates with a Done event.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import AsyncIterator, Awaitable, Callable

from harness.action_state import current_action_state, shape_tools_for_state
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
    preflight_tool_call,
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


def _append_preflight_feedback(conv: Conversation, name: str, message: str) -> Conversation:
    turns = conv.turns + (
        Turn("assistant", (TextPart(f"[tool preflight denied: {name}]"),)),
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


def _append_action_state_feedback(conv: Conversation, state: str, allowed: list[str]) -> Conversation:
    choices = ", ".join(allowed) or "a valid tool"
    turns = conv.turns + (
        Turn("assistant", (TextPart(f"[runtime action state requires tool: {state}]"),)),
        Turn("user", (TextPart(
            f"The current runtime action state is {state!r}; do not answer in free text yet. "
            f"Call one of these tools now: {choices}."
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


def _record_preflight(metrics: dict, call: ToolCall, decision) -> None:
    metrics["preflight_decision"] = decision.decision
    metrics["preflight_reason"] = decision.reason
    if decision.decision == "rewrite":
        metrics["preflight_rewrites"] += 1
    elif decision.decision == "deny":
        metrics["preflight_denies"] += 1
    if decision.reason:
        reasons = metrics.setdefault("preflight_reasons", {})
        reasons[decision.reason] = reasons.get(decision.reason, 0) + 1
    event = {
        "id": call.id,
        "tool": call.name,
        "decision": decision.decision,
        "reason": decision.reason,
        "original_arguments": decision.original_arguments,
        "rewritten_arguments": decision.rewritten_arguments,
        "bash_command_class": decision.bash_command_class,
    }
    metrics.setdefault("preflight_events", []).append(event)


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
    m.setdefault("preflight_decision", "none")
    m.setdefault("preflight_reason", None)
    m.setdefault("preflight_rewrites", 0)
    m.setdefault("preflight_denies", 0)
    m.setdefault("preflight_reasons", {})
    m.setdefault("preflight_events", [])
    m.setdefault("emitted_tool_calls", [])
    m.setdefault("invalid_tool_events", [])
    m.setdefault("action_state_blocks", 0)
    m.setdefault("first_attempt_constraints", 0)
    guard_metrics(m)
    attempts = 0
    suppress_text = False
    constraint_schema: dict | None = None
    constraint_tool_name: str | None = None
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
        action_state = current_action_state(conv, settings)
        m["action_state"] = action_state.name
        m["action_state_reason"] = action_state.reason
        state_available_tools = [
            tool.name for tool in conv.tools
            if not action_state.allowed_tools or tool.name in action_state.allowed_tools
        ]
        effective_requires_tool = action_state.requires_tool and bool(state_available_tools)
        payload_conv = shape_tools_for_state(conv, action_state)
        if constraint_tool_name and all(t.name != constraint_tool_name for t in payload_conv.tools):
            tool = next((t for t in conv.tools if t.name == constraint_tool_name), None)
            if tool is not None:
                payload_conv = replace(payload_conv, tools=payload_conv.tools + (tool,))
        m["allowed_tools"] = [tool.name for tool in payload_conv.tools]
        payload = profile.render(payload_conv, model_name)
        apply_reasoning_budget(payload, settings, backend, role, body or {}, payload_conv, m)
        required_tool = action_state.required_tool
        if effective_requires_tool and required_tool is None and len(payload_conv.tools) == 1:
            required_tool = payload_conv.tools[0].name
        if not attempts and backend.constrained and required_tool:
            required = next(
                (t for t in conv.tools if t.name == required_tool),
                None,
            )
            if required is not None:
                payload = backend.apply_constraint(payload, required.input_schema)
                m["first_attempt_constraints"] += 1
        if attempts and backend.constrained and constraint_schema is not None:
            payload = backend.apply_constraint(payload, constraint_schema)

        detector = DegenerateDetector()
        bad_call: ToolCall | None = None
        bad_error = ""
        loop_call: ToolCall | None = None
        loop_seen = 0
        emitted_valid_call = False
        guarded_call: tuple[str, str] | None = None
        preflight_feedback: tuple[str, str] | None = None
        action_state_feedback: tuple[str, list[str]] | None = None
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
        ) or effective_requires_tool

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
                    preflight = preflight_tool_call(conv, fixed, settings)
                    _record_preflight(m, fixed, preflight)
                    if preflight.decision == "rewrite":
                        fixed = preflight.call
                        if preflight.reason == "path_alias":
                            m["path_rewrites"] += 1
                            m["path_rewrite_names"].append(fixed.name)
                    elif preflight.decision == "deny":
                        if attempts < settings.pipeline.repair_retries:
                            preflight_feedback = (
                                preflight.reason or "denied",
                                preflight.feedback or "The tool call was denied by deterministic preflight.",
                            )
                            break
                        m["invalid_calls"] += 1
                        yield TextDelta(
                            f"\n[preflight denied {fixed.name}: {preflight.reason or 'denied'}]\n"
                        )
                        continue
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
                    m["emitted_tool_calls"].append({
                        "id": fixed.id,
                        "tool": fixed.name,
                        "arguments": fixed.arguments,
                    })
                    if ev.raw_arguments:  # arrived malformed, json-repaired locally
                        m["repaired_calls"] += 1
                    yield fixed
                elif attempts < settings.pipeline.repair_retries:
                    bad_call, bad_error = ev, error or "invalid"
                    raw_arguments = ev.raw_arguments or json.dumps(ev.arguments, separators=(",", ":"))
                    m["invalid_tool_events"].append({
                        "tool": ev.name,
                        "error": bad_error,
                        "arguments": ev.arguments,
                        "raw_arguments": raw_arguments,
                    })
                    break
                else:
                    m["invalid_calls"] += 1
                    raw = ev.raw_arguments or str(ev.arguments)
                    raw_arguments = ev.raw_arguments or json.dumps(ev.arguments, separators=(",", ":"))
                    m["invalid_tool_events"].append({
                        "tool": ev.name,
                        "error": bad_error or error,
                        "arguments": ev.arguments,
                        "raw_arguments": raw_arguments,
                    })
                    yield TextDelta(f"\n[invalid tool call {ev.name}: {bad_error or error}]\n{raw[:500]}")
            else:  # Done
                if buffered_text and not emitted_valid_call and ev.stop_reason != "tool_use":
                    guarded_done = guard_done_claim(conv, "".join(buffered_text), settings)
                    if guarded_done is not None and attempts < settings.pipeline.repair_retries:
                        break
                    if effective_requires_tool and attempts < settings.pipeline.repair_retries:
                        action_state_feedback = (action_state.name, [tool.name for tool in payload_conv.tools])
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

        if preflight_feedback is not None:
            attempts += 1
            guard, message = preflight_feedback
            suppress_text = True
            conv = _append_preflight_feedback(conv, guard, message)
            continue

        if action_state_feedback is not None:
            attempts += 1
            state_name, allowed = action_state_feedback
            suppress_text = True
            m["action_state_blocks"] += 1
            conv = _append_action_state_feedback(conv, state_name, allowed)
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
        constraint_tool_name = bad_call.name
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
