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
TARGET_RATIO = 0.8
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


def _truncate_results(turn: Turn) -> Turn:
    parts = tuple(
        replace(p, content=p.content[:HEAD] + ELISION + p.content[-TAIL:])
        if isinstance(p, ToolResultPart) and len(p.content) > TRUNCATE_OVER
        else p
        for p in turn.parts
    )
    return Turn(turn.role, parts)


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

    def apply(self, conv: Conversation, settings: Settings) -> Conversation:
        cw = settings.profile.context_window
        # Clients may request max_tokens larger than a small window; reserve
        # at most half the window for output so the budget never goes negative
        # (a negative budget evicts the whole conversation, task included).
        budget = cw - min(conv.params.max_tokens, cw // 2) - MARGIN
        if count_conversation(conv, self.counter) <= budget:
            return conv
        target = budget * TARGET_RATIO

        k = settings.pipeline.recent_turns_protected
        head, tail = conv.turns[:-k], conv.turns[-k:]

        # pass 1: truncate old tool results
        head = tuple(_truncate_results(t) for t in head)
        conv = replace(conv, turns=head + tail)
        if count_conversation(conv, self.counter) <= target:
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
        return conv
