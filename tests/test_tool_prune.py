from harness.config import Settings
from harness.ir import Conversation, GenParams, TextPart, ToolCallPart, ToolDef, Turn
from harness.pipeline.tool_prune import CORE, ToolPruneStage


def tool(name: str) -> ToolDef:
    return ToolDef(name, f"{name} tool", {"type": "object"}, {"type": "object"})


EXTRAS = ("WebSearch", "WebFetch", "NotebookEdit", "KillShell", "ListMcpResources",
          "ReadMcpResource", "ExitPlanMode")
ALL_TOOLS = tuple(tool(n) for n in CORE + EXTRAS)


def conv(turns=()) -> Conversation:
    return Conversation("s", tuple(turns), ALL_TOOLS, GenParams(max_tokens=100))


def test_core_kept_extras_dropped():
    out = ToolPruneStage().apply(conv(), Settings())
    names = {t.name for t in out.tools}
    assert names == set(CORE)
    assert len(out.tools) == 8


def test_recently_used_extra_kept():
    turns = (
        Turn("assistant", (ToolCallPart("t1", "WebFetch", {"url": "https://x"}),)),
        Turn("user", (TextPart("ok"),)),
    )
    out = ToolPruneStage().apply(conv(turns), Settings())
    names = [t.name for t in out.tools]
    assert "WebFetch" in names
    assert len(names) <= Settings().pipeline.max_tools


def test_old_usage_stays_surfaced():
    # Once a tool is called it must stay surfaced for the whole session:
    # dropping it later changes the rendered tool list, which rewrites the
    # prompt prefix and forces a full KV re-prefill (20-60s at 60k tokens).
    s = Settings()
    old = (Turn("assistant", (ToolCallPart("t1", "WebFetch", {"url": "u"}),)),)
    filler = tuple(
        Turn("user", (TextPart(f"msg {i}"),))
        for i in range(s.pipeline.recent_turns_protected + 1)
    )
    out = ToolPruneStage().apply(conv(old + filler), s)
    assert "WebFetch" in {t.name for t in out.tools}


def test_called_tools_keep_first_call_order():
    # First-call order is append-mostly: new calls extend the list at a
    # stable position instead of reshuffling it.
    turns = (
        Turn("assistant", (ToolCallPart("t1", "NotebookEdit", {}),)),
        Turn("user", (TextPart("ok"),)),
        Turn("assistant", (ToolCallPart("t2", "WebFetch", {"url": "u"}),)),
        Turn("user", (TextPart("ok"),)),
    )
    out = ToolPruneStage().apply(conv(turns), Settings())
    names = [t.name for t in out.tools]
    assert names.index("NotebookEdit") < names.index("WebFetch")


def test_toggle_off_identity():
    s = Settings()
    s.pipeline.tool_prune = False
    out = ToolPruneStage().apply(conv(), s)
    assert len(out.tools) == len(ALL_TOOLS)
    assert out.all_tools == ()  # prune off: everything surfaced, nothing hidden


def test_all_tools_records_full_inventory():
    # The relay surfaces hidden schemas from all_tools; pruning must
    # record the full inventory before cutting the surfaced set.
    out = ToolPruneStage().apply(conv(), Settings())
    assert out.all_tools == ALL_TOOLS
    assert len(out.tools) == Settings().pipeline.max_tools
    assert len(out.all_tools) == len(ALL_TOOLS)
