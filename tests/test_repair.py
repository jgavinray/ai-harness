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
    fixed, _ = repair_toolcall(call, TOOLS)
    assert fixed is None


EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string"},
        "old_string": {"type": "string"},
        "new_string": {"type": "string"},
        "replace_all": {"type": "boolean"},
    },
    "required": ["file_path", "old_string", "new_string"],
}
EDIT_TOOLS = (ToolDef("Edit", "edits", {"type": "object"}, EDIT_SCHEMA),)


def test_string_boolean_coerced():
    # observed live (envelope-27b rename-refactor): replace_all "false" as a
    # string was the only thing making the corrective Edit invalid
    call = ToolCall("t1", "Edit", {
        "file_path": "/x", "old_string": "a", "new_string": "b",
        "replace_all": "false",
    })
    fixed, err = repair_toolcall(call, EDIT_TOOLS)
    assert err is None
    assert fixed.arguments["replace_all"] is False


def test_wrapped_quotes_stripped_from_path_args():
    # live regression 2026-07-11 (session fee3e2f8): four Read calls went out
    # with file_path='"/home/.../CMakeLists.txt"' — literal quotes inside the
    # value — and failed client-side; the model noticed ("the Read tool seems
    # to be failing") and fell back to Bash cat. Only path-like keys are
    # stripped: content-bearing strings (old_string/new_string) can be
    # legitimately fully quoted code fragments.
    call = ToolCall("t1", "Read", {"file_path": '"/home/azeroth/x/CMakeLists.txt"'})
    fixed, err = repair_toolcall(call, TOOLS)
    assert err is None
    assert fixed.arguments["file_path"] == "/home/azeroth/x/CMakeLists.txt"


def test_quoted_content_strings_left_alone():
    call = ToolCall("t1", "Edit", {
        "file_path": "/x", "old_string": '"exact quoted fragment"',
        "new_string": '"new quoted fragment"',
    })
    fixed, err = repair_toolcall(call, EDIT_TOOLS)
    assert err is None
    assert fixed.arguments["old_string"] == '"exact quoted fragment"'


def test_string_number_coerced():
    call = ToolCall("t1", "Read", {"file_path": "/x", "limit": "25"})
    fixed, err = repair_toolcall(call, TOOLS)
    assert err is None
    assert fixed.arguments["limit"] == 25


def test_non_coercible_string_still_fails():
    call = ToolCall("t1", "Edit", {
        "file_path": "/x", "old_string": "a", "new_string": "b",
        "replace_all": "maybe",
    })
    fixed, err = repair_toolcall(call, EDIT_TOOLS)
    assert fixed is None
    assert "replace_all" in err or "boolean" in err or "maybe" in err


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
