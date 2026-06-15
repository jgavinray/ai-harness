"""Strong-model runtime review for risky executor checkpoints."""

from __future__ import annotations

import time

from harness.backends.pool import BackendPool, PooledBackend
from harness.config import Settings
from harness.ir import Conversation, TextDelta, ThinkingDelta, TextPart, ToolCallPart, ToolResultPart
from harness.log import RequestLogger
from harness.reasoning_budget import apply_reasoning_budget
from harness.tokens.counter import HeuristicCounter


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
    ) -> str | None:
        if not self.cfg.enabled or trigger not in self.cfg.triggers:
            return None
        backend = _review_backend(pool)
        if backend is None:
            metrics["review_skipped_no_backend"] = 1
            return None
        payload = {
            "model": backend.model_name,
            "messages": [
                {"role": "system", "content": _review_system()},
                {
                    "role": "user",
                    "content": _review_prompt(trigger, conv, default_feedback, self.cfg.max_chars),
                },
            ],
            "max_tokens": self.cfg.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        side_metrics: dict = {}
        apply_reasoning_budget(payload, self.settings, backend, "review", {}, conv, side_metrics)
        text = ""
        thinking = ""
        start = time.monotonic()
        try:
            async for ev in backend.profile.parse(backend.stream(payload)):
                if isinstance(ev, TextDelta):
                    text += ev.text
                elif isinstance(ev, ThinkingDelta):
                    thinking += ev.text
        except Exception as exc:
            metrics["review_error"] = str(exc)
            return None
        feedback = _feedback(text)
        metrics["review_trigger"] = trigger
        metrics["review_action"] = "revise" if feedback else "approve"
        metrics["review_reasoning_budget_sent"] = side_metrics.get("reasoning_budget_sent")
        metrics["review_reasoning_tokens_observed"] = HeuristicCounter().count_text(thinking) if thinking else 0
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
                "wall_ms": int((time.monotonic() - start) * 1000),
                "reasoning_tokens_observed": metrics["review_reasoning_tokens_observed"],
                **side_metrics,
            })
        if not feedback:
            return None
        metrics["review_generated"] = metrics.get("review_generated", 0) + 1
        return feedback


def _review_backend(pool: BackendPool) -> PooledBackend | None:
    candidates = pool.with_role("review") or pool.with_role("plan")
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
