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


MCP_TOOLS = tuple(
    tool(n) for n in ("mcp__github__create_pr", "mcp__github__list_issues",
                      "mcp__slack__send_message", "Skill")
)


def conv_mcp(text: str) -> Conversation:
    return Conversation(
        "s",
        (Turn("user", (TextPart(text),)),),
        ALL_TOOLS + MCP_TOOLS,
        GenParams(max_tokens=100),
    )


def test_named_mcp_server_surfaces_its_tools():
    # Regression for the CORE deadlock: CORE is 8 names, max_tools is 8,
    # so an MCP tool could never be surfaced no matter what the user said.
    out = ToolPruneStage().apply(conv_mcp("use the github mcp to open a PR"), Settings())
    names = {t.name for t in out.tools}
    assert "mcp__github__create_pr" in names
    assert "mcp__github__list_issues" in names
    assert "mcp__slack__send_message" not in names
    assert len(out.tools) <= Settings().pipeline.max_tools


def test_exact_tool_name_surfaces_tool():
    out = ToolPruneStage().apply(conv_mcp("call mcp__slack__send_message please"), Settings())
    assert "mcp__slack__send_message" in {t.name for t in out.tools}


def test_skill_mention_surfaces_skill_tool():
    out = ToolPruneStage().apply(conv_mcp("run the brainstorming skill"), Settings())
    assert "Skill" in {t.name for t in out.tools}


def test_tool_results_do_not_trigger_matching():
    # File contents flowing back through tool results must not surface
    # tools; only the user's own words count.
    from harness.ir import ToolResultPart
    turns = (
        Turn("user", (TextPart("fix the bug"),)),
        Turn("assistant", (ToolCallPart("t1", "Read", {"file_path": "/x"}),)),
        Turn("user", (ToolResultPart("t1", "docs mention mcp__slack__send_message here"),)),
    )
    out = ToolPruneStage().apply(
        Conversation("s", turns, ALL_TOOLS + MCP_TOOLS, GenParams(max_tokens=100)),
        Settings(),
    )
    assert "mcp__slack__send_message" not in {t.name for t in out.tools}
