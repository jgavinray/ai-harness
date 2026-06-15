"""Deliberate critic sidecar for risky executor turns.

Unlike ReviewManager, which only augments deterministic retry/guard feedback,
this runs before the next main executor action when the conversation contains
evidence of risky prior work.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from fnmatch import fnmatch

from harness.backends.pool import BackendPool, PooledBackend
from harness.config import Settings
from harness.ir import Conversation, Done, TextDelta, ThinkingDelta, TextPart, ToolCallPart, ToolResultPart, Turn
from harness.log import RequestLogger
from harness.reasoning_budget import apply_reasoning_budget
from harness.tokens.counter import HeuristicCounter

EDIT_TOOLS = {"Edit", "Write", "MultiEdit"}
BUILD_WORDS = ("gcc", "clang", "make", "cmake", "ninja", "ld ", "undefined reference")
TEST_WORDS = ("pytest", "failed", "failure", "assertion", "segfault", "asan", "ubsan")


class CriticManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cfg = settings.critic
        self.reviewed: dict[str, str] = {}

    async def maybe_inject(
        self,
        key: str,
        conv: Conversation,
        pool: BackendPool,
        metrics: dict,
        *,
        logger: RequestLogger | None = None,
        parent_request_id: str | None = None,
        account_usage=None,
    ) -> Conversation:
        if not self.cfg.enabled:
            return conv
        evidence = _evidence(conv, self.settings)
        metrics["critic_triggered"] = bool(evidence["triggers"])
        metrics["critic_triggers"] = evidence["triggers"]
        metrics["critic_matched_profiles"] = evidence["matched_profiles"]
        if not evidence["triggers"]:
            return conv
        fingerprint = _fingerprint(evidence)
        if self.reviewed.get(key) == fingerprint:
            metrics["critic_skipped_reason"] = "already_reviewed"
            return conv
        backend = _critic_backend(pool)
        if backend is None:
            metrics["critic_skipped_reason"] = "no_backend"
            return conv
        self.reviewed[key] = fingerprint
        payload = {
            "model": backend.model_name,
            "messages": [
                {"role": "system", "content": _critic_system()},
                {"role": "user", "content": _critic_prompt(conv, evidence, self.cfg.max_chars)},
            ],
            "max_tokens": self.cfg.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        side_metrics: dict = {}
        apply_reasoning_budget(payload, self.settings, backend, "critic", {}, conv, side_metrics)
        text = ""
        thinking = ""
        done = Done("end_turn")
        start = time.monotonic()
        try:
            async for ev in backend.profile.parse(backend.stream(payload)):
                if isinstance(ev, TextDelta):
                    text += ev.text
                elif isinstance(ev, ThinkingDelta):
                    thinking += ev.text
                elif isinstance(ev, Done):
                    done = ev
        except Exception as exc:
            metrics["critic_error"] = str(exc)
            return conv
        feedback = _feedback(text)
        action = "revise" if feedback else "approve"
        observed = HeuristicCounter().count_text(thinking) if thinking else 0
        metrics.update({
            "critic_action": action,
            "critic_reasoning_budget_sent": side_metrics.get("reasoning_budget_sent"),
            "critic_reasoning_tokens_observed": observed,
        })
        if account_usage:
            account_usage(backend, done, key, count_request=True)
        if logger:
            logger.write({
                "kind": "sidecar",
                "sidecar_type": "critic",
                "parent_request_id": parent_request_id,
                "session_key": key,
                "backend": backend.name,
                "model": backend.model_name,
                "role": "critic",
                "critic_action": action,
                "critic_triggers": evidence["triggers"],
                "critic_matched_profiles": evidence["matched_profiles"],
                "wall_ms": int((time.monotonic() - start) * 1000),
                "input_tokens": done.input_tokens,
                "output_tokens": done.output_tokens,
                "cached_tokens": done.cached_tokens,
                "stop_reason": done.stop_reason,
                "reasoning_tokens_observed": observed,
                **side_metrics,
            })
        if not feedback:
            return conv
        metrics["critic_generated"] = metrics.get("critic_generated", 0) + 1
        return replace(conv, turns=conv.turns + (Turn("user", (TextPart(f"Critic feedback:\n{feedback}"),)),))


def _critic_backend(pool: BackendPool) -> PooledBackend | None:
    candidates = pool.with_role("critic") or pool.with_role("review") or pool.with_role("plan")
    if not candidates:
        return None
    return min(candidates, key=lambda b: (b.in_flight, b.requests))


def _evidence(conv: Conversation, settings: Settings) -> dict:
    triggers: list[str] = []
    paths: list[str] = []
    tool_calls: list[str] = []
    tool_errors = 0
    result_texts: list[str] = []
    for turn in conv.turns[-12:]:
        for part in turn.parts:
            if isinstance(part, ToolCallPart):
                tool_calls.append(part.name)
                for value in part.arguments.values():
                    if isinstance(value, str):
                        paths.extend(_pathish(value))
                if part.name in EDIT_TOOLS and "edit" in settings.critic.triggers:
                    triggers.append("edit")
            elif isinstance(part, ToolResultPart):
                result_texts.append(part.content[:1000])
                if part.is_error and "tool_error" in settings.critic.triggers:
                    tool_errors += 1
                    triggers.append("tool_error")
                lower = part.content.lower()
                if any(word in lower for word in BUILD_WORDS) and "build_failure" in settings.critic.triggers:
                    triggers.append("build_failure")
                if any(word in lower for word in TEST_WORDS) and "test_failure" in settings.critic.triggers:
                    triggers.append("test_failure")
    matched_profiles: list[str] = []
    all_text = "\n".join(result_texts + paths).lower()
    for profile in settings.risk_profiles:
        path_match = any(fnmatch(path.removeprefix("./"), pattern) for path in paths for pattern in profile.path_patterns)
        text_match = any(pattern.lower() in all_text for pattern in profile.text_patterns)
        if path_match or text_match:
            matched_profiles.append(profile.name)
            if "risky_path" in settings.critic.triggers:
                triggers.append("risky_path")
    return {
        "triggers": sorted(set(triggers)),
        "paths": sorted(set(paths))[:40],
        "tool_calls": tool_calls[-20:],
        "tool_errors": tool_errors,
        "matched_profiles": matched_profiles,
        "recent_results": result_texts[-6:],
    }


def _pathish(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.replace("'", " ").replace('"', " ").split():
        token = raw.strip(".,:;()[]{}<>`")
        if "/" in token or token.endswith((".c", ".h", ".cc", ".cpp", ".hpp", ".rs", ".py")):
            out.append(token)
    return out


def _fingerprint(evidence: dict) -> str:
    return hashlib.sha1(json.dumps(evidence, sort_keys=True).encode()).hexdigest()


def _critic_system() -> str:
    return (
        "You are a senior critic for a coding agent. Review risky recent work "
        "before the executor continues. Return APPROVE if no corrective action "
        "is needed. Otherwise return concise, specific blockers and fixes. "
        "Do not expose chain-of-thought."
    )


def _critic_prompt(conv: Conversation, evidence: dict, max_chars: int) -> str:
    history: list[str] = []
    for turn in conv.turns[-10:]:
        parts: list[str] = []
        for part in turn.parts:
            if isinstance(part, TextPart):
                parts.append(part.text)
            elif isinstance(part, ToolCallPart):
                parts.append(f"[tool call: {part.name} {part.arguments}]")
            elif isinstance(part, ToolResultPart):
                marker = " ERROR" if part.is_error else ""
                parts.append(f"[tool result{marker}: {part.content[:1200]}]")
        if parts:
            history.append(f"{turn.role}: " + "\n".join(parts))
    body = (
        f"Triggers: {', '.join(evidence['triggers'])}\n"
        f"Matched profiles: {', '.join(evidence['matched_profiles']) or 'none'}\n"
        f"Paths: {', '.join(evidence['paths']) or 'none'}\n"
        f"Tool calls: {', '.join(evidence['tool_calls']) or 'none'}\n\n"
        f"System/plan context:\n{conv.system[-max_chars // 4:]}\n\n"
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
    return cleaned[:1200]
