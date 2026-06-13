"""Stage ②: prune the tool list so small models see at most max_tools.

Priority: tools called anywhere in history (sticky, first-call order, for
KV-prefix stability), then the core set, then everything else.
"""

from __future__ import annotations

from dataclasses import replace

from harness.config import Settings
from harness.ir import Conversation, ToolCallPart

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

        by_name = {t.name: t for t in conv.tools}
        keep: list[str] = []
        for name in (*called, *CORE, *by_name):
            if name in by_name and name not in keep:
                keep.append(name)
            if len(keep) >= settings.pipeline.max_tools:
                break
        return replace(
            conv, tools=tuple(by_name[n] for n in keep), all_tools=conv.tools
        )
