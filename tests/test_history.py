from harness.config import Settings
from harness.ir import (
    Conversation,
    GenParams,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    Turn,
)
from harness.pipeline.history import ELISION, HistoryStage


def big_session(n_pairs: int = 12, result_size: int = 6000) -> Conversation:
    turns = [Turn("user", (TextPart("fix the bug"),))]
    for i in range(n_pairs):
        turns.append(
            Turn("assistant", (ToolCallPart(f"t{i}", "Read", {"file_path": f"/f{i}"}),))
        )
        turns.append(Turn("user", (ToolResultPart(f"t{i}", f"chunk{i} " + "x" * result_size),)))
    return Conversation("system", tuple(turns), (), GenParams(max_tokens=1024))


def small_settings(window: int) -> Settings:
    s = Settings()
    s.profile.context_window = window
    return s


def test_under_budget_identity():
    conv = big_session(2, 100)
    out = HistoryStage().apply(conv, small_settings(32768))
    assert out is conv


def test_old_results_truncated_recent_protected():
    conv = big_session()
    s = small_settings(16000)
    out = HistoryStage().apply(conv, s)
    k = s.pipeline.recent_turns_protected
    assert out.turns[-k:] == conv.turns[-k:]  # protected tail byte-identical
    old_results = [
        p for t in out.turns[:-k] for p in t.parts if isinstance(p, ToolResultPart)
    ]
    assert any(ELISION in p.content for p in old_results)
    assert out.system == conv.system


def test_eviction_keeps_pairing():
    conv = big_session(20, 8000)
    out = HistoryStage().apply(conv, small_settings(4000))
    # invariant: every surviving tool result's call id exists in a surviving turn
    call_ids = {
        p.id for t in out.turns for p in t.parts if isinstance(p, ToolCallPart)
    }
    for t in out.turns:
        for p in t.parts:
            if isinstance(p, ToolResultPart):
                assert p.tool_call_id in call_ids
    # eviction marker present
    assert any(
        isinstance(p, TextPart) and "elided" in p.text
        for t in out.turns
        for p in t.parts
    )
    assert len(out.turns) < len(conv.turns)
