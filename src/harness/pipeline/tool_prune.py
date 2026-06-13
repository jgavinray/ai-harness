"""Stage ②: prune the tool list so small models see at most max_tools.

Priority: tools called anywhere in history (sticky, first-call order, for
KV-prefix stability), then the core set, then everything else.
"""

from __future__ import annotations

import re
from dataclasses import replace

from harness.config import Settings
from harness.ir import Conversation, TextPart, ToolCallPart


def _last_user_text(conv: Conversation) -> str:
    """The newest user turn that contains actual user words (TextPart).
    Tool-result-only turns are skipped so file contents can't trigger
    matches, and the match stays stable until the user speaks again."""
    for turn in reversed(conv.turns):
        if turn.role != "user":
            continue
        texts = [p.text for p in turn.parts if isinstance(p, TextPart)]
        if texts:
            return "\n".join(texts).lower()
    return ""


def _mentioned(word: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(word.lower())}s?\b", text) is not None


def _named_tools(conv: Conversation) -> list[str]:
    text = _last_user_text(conv)
    if not text:
        return []
    named: list[str] = []
    for t in conv.tools:
        if _mentioned(t.name, text):
            named.append(t.name)
        elif t.name.startswith("mcp__"):
            server = t.name.split("__")[1]
            if _mentioned(server, text):
                named.append(t.name)
    return named


CATALOG_HEADER = (
    "## Tool catalog\n"
    "You may call ANY tool below by name, even if its full schema is not "
    "provided; the schema will be supplied when you use it."
)


def _summary(description: str) -> str:
    first = description.strip().split("\n", 1)[0]
    first = first.split(". ", 1)[0].rstrip(".")
    return first[:80]


def _catalog(tools: tuple) -> str:
    lines = [f"- {t.name} — {_summary(t.description)}" for t in tools]
    return CATALOG_HEADER + "\n" + "\n".join(lines)


CORE = ("Read", "Edit", "Write", "Bash", "Grep", "Glob", "TodoWrite", "Task")


class ToolPruneStage:
    def apply(self, conv: Conversation, settings: Settings) -> Conversation:
        if not settings.pipeline.tool_prune or not conv.tools:
            return conv

        called: list[str] = []
        for turn in conv.turns:
            for part in turn.parts:
                if isinstance(part, ToolCallPart) and part.name not in called:
                    called.append(part.name)

        named = _named_tools(conv)
        by_name = {t.name: t for t in conv.tools}
        keep: list[str] = []
        for name in (*called, *named, *CORE, *by_name):
            if name in by_name and name not in keep:
                keep.append(name)
            if len(keep) >= settings.pipeline.max_tools:
                break
        # Soft cap: user-named tools always surface, appended at the END so
        # the called-tool prefix stays byte-stable. Bounded by max_tools so
        # a huge server alias can at worst double the list, and the set
        # shrinks back once a later user message stops naming them.
        for name in named[: settings.pipeline.max_tools]:
            if name in by_name and name not in keep:
                keep.append(name)
        system = conv.system
        if settings.pipeline.tool_catalog:
            system = system + "\n\n" + _catalog(conv.tools)
        return replace(
            conv,
            tools=tuple(by_name[n] for n in keep),
            all_tools=conv.tools,
            system=system,
        )
