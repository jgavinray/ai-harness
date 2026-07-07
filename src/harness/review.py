"""Strong-model runtime review for risky executor checkpoints."""

from __future__ import annotations

import time

from harness.backends.pool import BackendPool, PooledBackend
from harness.config import Settings
from harness.ir import Conversation, Done, TextDelta, ThinkingDelta, TextPart, ToolCallPart, ToolResultPart
from harness.log import RequestLogger
from harness.reasoning_budget import apply_reasoning_budget
from harness.tokens.counter import HeuristicCounter

CRITIC_RUNTIME_TRIGGERS = {
    "loop_break",
    "missing_parent",
    "missing_parent_next_action",
    "use_write_tool",
    "repeated_failing_call",
    "use_read_tool",
    "use_grep_tool",
    "dangerous_command",
    "non_verification_command",
    "invalid_tool_retry",
    "plan_drift",
    "verify_after_edit",
}


class ReviewManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cfg = settings.review

    async def review(
        self,
        trigger: str,
        conv: Conversation,
        default_feedback: str,
        pool: BackendPool,
        metrics: dict,
        *,
        logger: RequestLogger | None = None,
        parent_request_id: str | None = None,
        session_key: str | None = None,
        account_usage=None,
    ) -> str | None:
        enabled = self.cfg.enabled or self.settings.critic.enabled
        triggers = set(self.cfg.triggers)
        if self.settings.critic.enabled:
            triggers.update(CRITIC_RUNTIME_TRIGGERS)
        if not enabled or trigger not in triggers:
            return None
        backend = _review_backend(pool)
        if backend is None:
            metrics["review_skipped_no_backend"] = 1
            return None
        max_tokens = self.cfg.max_tokens
        if self.settings.critic.enabled:
            max_tokens = max(max_tokens, self.settings.critic.max_tokens)
        payload = {
            "model": backend.model_name,
            "messages": [
                {"role": "system", "content": _review_system()},
                {
                    "role": "user",
                    "content": _review_prompt(trigger, conv, default_feedback, self.cfg.max_chars),
                },
            ],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        side_metrics: dict = {}
        apply_reasoning_budget(payload, self.settings, backend, "review", {}, conv, side_metrics)
        text = ""
        thinking = ""
        done = Done("end_turn")
        ttft_ms: int | None = None
        start = time.monotonic()
        try:
            async for ev in backend.profile.parse(backend.stream(payload)):
                if ttft_ms is None:
                    ttft_ms = int((time.monotonic() - start) * 1000)
                if isinstance(ev, TextDelta):
                    text += ev.text
                elif isinstance(ev, ThinkingDelta):
                    thinking += ev.text
                elif isinstance(ev, Done):
                    done = ev
        except Exception as exc:
            metrics["review_error"] = str(exc)
            return None
        feedback = _feedback(text)
        inconclusive_reason = _inconclusive_reason(text, done)
        metrics["review_trigger"] = trigger
        metrics["review_action"] = "revise" if feedback else ("inconclusive" if inconclusive_reason else "approve")
        metrics["review_reasoning_budget_sent"] = side_metrics.get("reasoning_budget_sent")
        metrics["review_reasoning_tokens_observed"] = HeuristicCounter().count_text(thinking) if thinking else 0
        if inconclusive_reason:
            metrics["review_inconclusive_reason"] = inconclusive_reason
        if account_usage:
            account_usage(backend, done, session_key, count_request=True, ttft_ms=ttft_ms)
        if logger:
            logger.write({
                "kind": "sidecar",
                "sidecar_type": "review",
                "parent_request_id": parent_request_id,
                "session_key": session_key,
                "backend": backend.name,
                "model": backend.model_name,
                "role": "review",
                "review_trigger": trigger,
                "review_action": metrics["review_action"],
                "review_inconclusive_reason": inconclusive_reason,
                "wall_ms": int((time.monotonic() - start) * 1000),
                "ttft_ms": ttft_ms,
                "input_tokens": done.input_tokens,
                "output_tokens": done.output_tokens,
                "cached_tokens": done.cached_tokens,
                "stop_reason": done.stop_reason,
                "reasoning_tokens_observed": metrics["review_reasoning_tokens_observed"],
                **side_metrics,
            })
        if not feedback and inconclusive_reason:
            metrics["review_generated"] = metrics.get("review_generated", 0) + 1
            return (
                "The runtime critic hit its token limit without a usable decision. "
                "Follow the deterministic guard feedback above exactly; do not repeat "
                "the denied action."
            )
        if not feedback:
            return None
        metrics["review_generated"] = metrics.get("review_generated", 0) + 1
        return feedback


def _review_backend(pool: BackendPool) -> PooledBackend | None:
    candidates = pool.with_role("review") or pool.with_role("critic") or pool.with_role("plan")
    if not candidates:
        return None
    return min(candidates, key=lambda b: (b.in_flight, b.requests))


def _review_system() -> str:
    return (
        "You are a runtime critic for a coding agent. Decide whether the "
        "agent's next action is risky. Return one concise corrective message "
        "only when revision is needed. Do not expose chain-of-thought."
    )


def _review_prompt(
    trigger: str, conv: Conversation, default_feedback: str, max_chars: int
) -> str:
    history: list[str] = []
    for turn in conv.turns[-8:]:
        parts: list[str] = []
        for part in turn.parts:
            if isinstance(part, TextPart):
                parts.append(part.text)
            elif isinstance(part, ToolCallPart):
                parts.append(f"[tool call: {part.name} {part.arguments}]")
            elif isinstance(part, ToolResultPart):
                parts.append(f"[tool result: {part.content[:500]}]")
        if parts:
            history.append(f"{turn.role}: " + "\n".join(parts))
    body = (
        f"Trigger: {trigger}\n\n"
        f"Existing deterministic feedback:\n{default_feedback}\n\n"
        f"System/plan context:\n{conv.system[-max_chars // 3:]}\n\n"
        f"Recent conversation:\n" + "\n\n".join(history)
    )
    return body[-max_chars:]


def _feedback(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    if lowered.startswith("approve") or lowered.startswith("no-op") or lowered == "ok":
        return ""
    return cleaned[:600]


def _inconclusive_reason(text: str, done: Done) -> str | None:
    if done.stop_reason != "max_tokens":
        return None
    if _feedback(text):
        return None
    return "max_tokens_without_feedback"
