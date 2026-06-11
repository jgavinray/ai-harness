from harness.ir import ToolCall, ToolDef
from harness.repair.degenerate import DegenerateDetector
from harness.repair.toolcalls import repair_toolcall

READ_SCHEMA = {
    "type": "object",
    "properties": {"file_path": {"type": "string"}, "limit": {"type": "number"}},
    "required": ["file_path"],
}
TOOLS = (ToolDef("Read", "reads", {"type": "object"}, READ_SCHEMA),)


def test_valid_call_unchanged():
    call = ToolCall("t1", "Read", {"file_path": "/x"})
    fixed, err = repair_toolcall(call, TOOLS)
    assert err is None and fixed == call


def test_trailing_comma_repaired():
    call = ToolCall("t1", "Read", {}, raw_arguments='{"file_path": "/x",}')
    fixed, err = repair_toolcall(call, TOOLS)
    assert err is None
    assert fixed.arguments == {"file_path": "/x"}
    assert fixed.raw_arguments == ""


def test_missing_required_param():
    call = ToolCall("t1", "Read", {"limit": 3})
    fixed, err = repair_toolcall(call, TOOLS)
    assert fixed is None
    assert "file_path" in err


def test_unknown_tool():
    call = ToolCall("t1", "Wat", {"x": 1})
    fixed, err = repair_toolcall(call, TOOLS)
    assert fixed is None
    assert "Read" in err  # error lists available tools


def test_unrepairable_garbage():
    call = ToolCall("t1", "Read", {}, raw_arguments="not json at all {{{")
    fixed, err = repair_toolcall(call, TOOLS)
    assert fixed is None and err


def test_degenerate_detects_repetition():
    det = DegenerateDetector()
    tripped = False
    for _ in range(200):
        if det.feed("abc def "):
            tripped = True
            break
    assert tripped


def test_normal_prose_not_flagged():
    det = DegenerateDetector()
    prose = (
        "The relay loop validates each tool call against the original schema. "
        "When validation fails it appends feedback and retries the backend. "
        "Different sentences avoid periodic structure in this stream of text. "
        "Numbers like 1, 22, 333 and names like alpha, beta, gamma vary too. "
    )
    assert not any(det.feed(w + " ") for w in (prose * 3).split())
