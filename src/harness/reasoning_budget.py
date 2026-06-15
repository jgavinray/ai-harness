"""Reasoning-token budget policy for capable backends.

The policy is intentionally model-agnostic. A backend opts in with the
``reasoning_budget`` capability, and this module decides whether to add the
vLLM-compatible ``thinking_token_budget`` request field.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any

from harness.config import Settings
from harness.ir import Conversation, TextPart, ToolCallPart, ToolResultPart

MANUAL_DEEP_SIGNALS = (
    "32k",
    "deep reasoning",
    "think deeply",
    "exhaustive",
    "exhaustively",
    "comprehensive analysis",
)

ARCH_SIGNALS = (
    "architecture",
    "architectural",
    "design",
    "refactor",
    "migration",
    "tradeoff",
    "trade-off",
)

DEBUG_SIGNALS = (
    "debug",
    "diagnose",
    "root cause",
    "failure",
    "failing",
    "integration",
)


@dataclass(frozen=True)
class ReasoningDecision:
    enabled: bool
    budget: int | None
    mode: str
    source: str
    skipped_reason: str | None
    matched_profiles: list[str]
    signals: list[str]
    clamped_by: str | None


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict)
        )
    return ""


def _latest_user_text(body: dict) -> str:
    for msg in reversed(body.get("messages") or []):
        if msg.get("role") == "user":
            return _flatten_content(msg.get("content"))
    return ""


def _conversation_text(conv: Conversation) -> str:
    parts: list[str] = [conv.system]
    for turn in conv.turns:
        for part in turn.parts:
            if isinstance(part, TextPart):
                parts.append(part.text)
            elif isinstance(part, ToolCallPart):
                parts.extend(str(v) for v in part.arguments.values())
            elif isinstance(part, ToolResultPart):
                parts.append(part.content)
    return "\n".join(parts)


def _body_text(body: dict) -> str:
    parts: list[str] = [_flatten_content(body.get("system") or "")]
    for msg in body.get("messages") or []:
        parts.append(_flatten_content(msg.get("content")))
    return "\n".join(parts)


def _pathish_tokens(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.replace("'", " ").replace('"', " ").split():
        token = raw.strip(".,:;()[]{}<>`")
        if "/" in token or token.endswith((".c", ".h", ".cc", ".cpp", ".hpp", ".rs", ".py")):
            out.append(token.removeprefix("./"))
    return out


def _profile_matches(patterns: list[str], values: list[str]) -> bool:
    for pattern in patterns:
        for value in values:
            if fnmatch(value, pattern):
                return True
    return False


def _base_mode(role: str) -> str:
    if role == "plan":
        return "architecture_plan"
    if role in {"review", "critic"}:
        return "critic"
    if role == "reasoning":
        return "reasoning"
    return "default"


def decide(settings: Settings, backend: Any, role: str, body: dict, conv: Conversation) -> ReasoningDecision:
    cfg = settings.reasoning_budget
    if not cfg.enabled:
        return ReasoningDecision(False, None, "disabled", "disabled", "disabled", [], [], None)

    capabilities = set(getattr(getattr(backend, "cfg", backend), "capabilities", []) or [])
    if "reasoning_budget" not in capabilities:
        return ReasoningDecision(
            True, None, "unavailable", "capability", "backend_lacks_capability", [], [], None
        )

    text = "\n".join([_body_text(body), _conversation_text(conv)])
    text_l = text.lower()
    paths = _pathish_tokens(text)
    signals: list[str] = []
    matched_profiles: list[str] = []

    mode = _base_mode(role)
    source = "role"
    budget = cfg.role_tokens.get(role, cfg.default_tokens)

    for profile in settings.risk_profiles:
        path_match = _profile_matches(profile.path_patterns, paths)
        text_match = any(p.lower() in text_l for p in profile.text_patterns)
        if path_match or text_match:
            matched_profiles.append(profile.name)
            if path_match:
                signals.append(f"profile_path:{profile.name}")
            if text_match:
                signals.append(f"profile_text:{profile.name}")
            profile_mode = profile.critic_mode if role in {"review", "critic"} else profile.plan_mode
            if profile_mode and profile_mode in cfg.mode_tokens:
                mode = profile_mode
                budget = cfg.mode_tokens[profile_mode]
                source = "risk_profile"

    if any(s in text_l for s in DEBUG_SIGNALS) and cfg.mode_tokens.get("integration_debug", 0) > budget:
        signals.append("debug_keyword")
        mode = "integration_debug"
        budget = cfg.mode_tokens[mode]
        source = "keyword"
    if any(s in text_l for s in ARCH_SIGNALS) and cfg.mode_tokens.get("architecture_plan", 0) > budget:
        signals.append("architecture_keyword")
        mode = "architecture_plan"
        budget = cfg.mode_tokens[mode]
        source = "keyword"

    manual = any(s in text_l for s in MANUAL_DEEP_SIGNALS)
    if manual:
        signals.append("manual_deep_signal")

    clamped_by: str | None = None
    max_policy = cfg.max_manual_tokens if manual else cfg.max_auto_tokens
    if budget > max_policy:
        budget = max_policy
        clamped_by = "max_manual_tokens" if manual else "max_auto_tokens"

    max_tokens = conv.params.max_tokens
    output_safe = max_tokens - cfg.final_answer_reserve
    if output_safe <= 0:
        output_safe = max(max_tokens // 2, 0)
    if output_safe <= 0:
        return ReasoningDecision(
            True, None, mode, source, "no_output_budget", matched_profiles, signals, "final_answer_reserve"
        )
    if budget > output_safe:
        budget = output_safe
        clamped_by = "final_answer_reserve"

    if cfg.load_shed and getattr(backend, "in_flight", 0) > 0 and budget > cfg.default_tokens:
        budget = max(cfg.default_tokens, budget // 2)
        clamped_by = "load_shed"
        signals.append("backend_in_flight")

    if budget <= 0:
        return ReasoningDecision(True, None, mode, source, "non_positive_budget", matched_profiles, signals, clamped_by)

    return ReasoningDecision(True, int(budget), mode, source, None, matched_profiles, signals, clamped_by)


def apply_reasoning_budget(
    payload: dict,
    settings: Settings,
    backend: Any,
    role: str,
    body: dict,
    conv: Conversation,
    metrics: dict,
) -> ReasoningDecision:
    decision = decide(settings, backend, role, body, conv)
    if decision.budget is not None:
        payload["thinking_token_budget"] = decision.budget
    metrics.update(
        {
            "reasoning_budget_enabled": decision.enabled,
            "reasoning_budget_sent": decision.budget,
            "reasoning_budget_mode": decision.mode,
            "reasoning_budget_source": decision.source,
            "reasoning_budget_skipped_reason": decision.skipped_reason,
            "reasoning_budget_clamped_by": decision.clamped_by,
            "matched_risk_profiles": decision.matched_profiles,
            "reasoning_signals": decision.signals,
            "final_answer_reserve": settings.reasoning_budget.final_answer_reserve,
        }
    )
    return decision
