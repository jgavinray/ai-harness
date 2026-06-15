"""Stage ④: deterministic context-budget compaction.

Two passes when over budget:
  1. truncate old tool results to head+tail with an elision marker;
  2. evict whole turn-groups from the front (an assistant turn travels
     with the user turn that carries its tool results, so no orphan
     tool_result ever survives its call).
System prompt and the protected recent tail are never modified.
"""

from __future__ import annotations

from dataclasses import replace

from harness.config import Settings
from harness.ir import Conversation, TextPart, ToolCallPart, ToolResultPart, Turn
from harness.tokens.counter import HeuristicCounter, count_conversation

ELISION = "\n…[elided by harness]…\n"
HEAD, TAIL, TRUNCATE_OVER = 800, 300, 1500
MARGIN = 1024
# When compaction triggers, compact down to this fraction of the budget so
# the next few turns don't immediately re-trigger it (rewriting old turns
# every turn would invalidate the backend's KV prefix cache each request).
# 0.8 left only ~20% headroom: saturated sessions re-compacted every few
# turns and re-prefilled their whole context each time (observed 21s TTFT
# p50 at ~60k tokens). Half the budget keeps the prefix byte-stable ~2.5x
# longer between full re-prefills.
TARGET_RATIO = 0.5
DIGEST_MAX_TOOLS = 8


def _digest(evicted: list[tuple[Turn, ...]]) -> Turn:
    """Deterministic summary of evicted groups: byte-stable between
    compaction events so the backend KV prefix is only invalidated when
    eviction itself changes, and zero-latency (no LLM call)."""
    n_turns = sum(len(g) for g in evicted)
    tools: list[str] = []
    for g in evicted:
        for t in g:
            for p in t.parts:
                if isinstance(p, ToolCallPart) and p.name not in tools:
                    tools.append(p.name)
    used = ", ".join(tools[:DIGEST_MAX_TOOLS]) or "no tools"
    text = (
        f"[{n_turns} earlier turns elided by harness; tools used: {used}. "
        "Results of that work appear in later turns.]"
    )
    return Turn("user", (TextPart(text),))


def _truncate_results(turn: Turn) -> tuple[Turn, int]:
    truncated = 0
    parts = tuple(
        (
            replace(p, content=p.content[:HEAD] + ELISION + p.content[-TAIL:])
            if isinstance(p, ToolResultPart) and len(p.content) > TRUNCATE_OVER
            else p
        )
        for p in turn.parts
    )
    for old, new in zip(turn.parts, parts):
        if old is not new and isinstance(old, ToolResultPart):
            truncated += 1
    return Turn(turn.role, parts), truncated


def _groups(turns: tuple[Turn, ...]) -> list[tuple[Turn, ...]]:
    """Split into evictable units: assistant turn + following user turn
    holding its tool results; standalone turns are their own group."""
    groups: list[tuple[Turn, ...]] = []
    i = 0
    while i < len(turns):
        turn = turns[i]
        if (
            turn.role == "assistant"
            and i + 1 < len(turns)
            and any(isinstance(p, ToolResultPart) for p in turns[i + 1].parts)
        ):
            groups.append((turn, turns[i + 1]))
            i += 2
        else:
            groups.append((turn,))
            i += 1
    return groups


class HistoryStage:
    def __init__(self) -> None:
        self.counter = HeuristicCounter()

    def apply(
        self, conv: Conversation, settings: Settings, metrics: dict | None = None
    ) -> Conversation:
        cw = settings.profile.context_window
        # Clients may request max_tokens larger than a small window; reserve
        # at most half the window for output so the budget never goes negative
        # (a negative budget evicts the whole conversation, task included).
        hard_budget = cw - min(conv.params.max_tokens, cw // 2) - MARGIN
        effective_window = settings.pipeline.effective_context_window or cw
        threshold = min(
            hard_budget,
            int(effective_window * settings.pipeline.compact_at_ratio),
        )
        target = min(
            int(hard_budget * settings.pipeline.compact_target_ratio),
            int(effective_window * settings.pipeline.compact_target_ratio),
        )
        before = count_conversation(conv, self.counter)
        if metrics is not None:
            metrics.update({
                "context_tokens_before": before,
                "context_tokens_after": before,
                "context_budget": threshold,
                "context_effective_window": effective_window,
                "context_compacted": False,
                "compaction_reason": None,
                "turns_elided": 0,
                "tool_results_truncated": 0,
            })
        if before <= threshold:
            return conv

        k = settings.pipeline.recent_turns_protected
        head, tail = (conv.turns[:-k], conv.turns[-k:]) if k else (conv.turns, ())

        # pass 1: truncate old tool results
        truncated = 0
        truncated_head = []
        for turn in head:
            new_turn, n = _truncate_results(turn)
            truncated += n
            truncated_head.append(new_turn)
        head = tuple(truncated_head)
        conv = replace(conv, turns=head + tail)
        after_truncate = count_conversation(conv, self.counter)
        if after_truncate <= target:
            if metrics is not None:
                metrics.update({
                    "context_tokens_after": after_truncate,
                    "context_compacted": after_truncate < before,
                    "compaction_reason": "tool_result_truncation",
                    "tool_results_truncated": truncated,
                })
            return conv

        # pass 2: evict turn-groups from the front, but never the anchor —
        # the opening user turn that states the task.
        anchor: tuple[Turn, ...] = ()
        if head and head[0].role == "user" and any(
            isinstance(p, TextPart) for p in head[0].parts
        ):
            anchor, head = head[:1], head[1:]
        groups = _groups(head)
        dropped: list[tuple[Turn, ...]] = []
        while groups and count_conversation(conv, self.counter) > target:
            dropped.append(groups.pop(0))
            kept = tuple(t for g in groups for t in g)
            marker = (_digest(dropped),) if dropped else ()
            conv = replace(conv, turns=anchor + marker + kept + tail)
        after = count_conversation(conv, self.counter)
        if metrics is not None:
            metrics.update({
                "context_tokens_after": after,
                "context_compacted": after < before,
                "compaction_reason": "turn_eviction" if dropped else "tool_result_truncation",
                "turns_elided": sum(len(g) for g in dropped),
                "tool_results_truncated": truncated,
            })
        return conv
