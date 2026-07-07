from harness.action_state import current_action_state, shape_tools_for_state
from harness.config import Settings
from harness.ir import (
    Conversation,
    GenParams,
    TextPart,
    ToolCallPart,
    ToolDef,
    ToolResultPart,
    Turn,
)

MULTI_STEP_BRIEF = (
    "Running `python3 test_pipeline.py` fails. Diagnose why, fix the project "
    "so the tests pass, and verify. You may create new files if a module is "
    "missing, but do not change test_pipeline.py."
)


def _tools():
    schema = {"type": "object"}
    return tuple(ToolDef(n, n, schema, schema) for n in ("Read", "Bash", "Edit", "Write"))


def test_verify_request_surfaces_bash():
    read = ToolDef("Read", "reads", {"type": "object"}, {"type": "object"})
    bash = ToolDef("Bash", "runs", {"type": "object"}, {"type": "object"})
    conv = Conversation(
        "sys",
        (Turn("user", (TextPart("run tests"),)),),
        (read, bash),
        GenParams(max_tokens=512),
    )
    state = current_action_state(conv, Settings())
    shaped = shape_tools_for_state(conv, state)
    assert state.name == "verify"
    assert [tool.name for tool in shaped.tools] == ["Read", "Bash"]
    assert state.required_tool == "Bash"


def test_task_brief_mentioning_verify_keeps_workspace_tools():
    # eval regression (brick1-verify, multi-step 0/5): a long task brief that
    # mentions "verify" must not lock the session into verify state, and file
    # creation via Write must stay possible after reading.
    conv = Conversation(
        "sys",
        (
            Turn("user", (TextPart(MULTI_STEP_BRIEF),)),
            Turn("assistant", (ToolCallPart("r1", "Read", {"file_path": "/x/test_pipeline.py"}),)),
            Turn("user", (ToolResultPart("r1", "import textutil"),)),
        ),
        _tools(),
        GenParams(max_tokens=512),
    )
    state = current_action_state(conv, Settings())
    assert state.name == "edit_existing"
    assert "Write" in state.allowed_tools
    assert "Edit" in state.allowed_tools


def test_long_brief_with_create_words_does_not_lock_create_state():
    conv = Conversation(
        "sys",
        (Turn("user", (TextPart(MULTI_STEP_BRIEF),)),),
        _tools(),
        GenParams(max_tokens=512),
    )
    state = current_action_state(conv, Settings())
    assert state.name == "inspect"
    assert "Write" in state.allowed_tools
    assert "Read" in state.allowed_tools


def test_effort_testing_text_does_not_force_verify():
    read = ToolDef("Read", "reads", {"type": "object"}, {"type": "object"})
    bash = ToolDef("Bash", "runs", {"type": "object"}, {"type": "object"})
    conv = Conversation(
        "sys",
        (Turn("user", (TextPart("Set effort level to high: Comprehensive implementation with extensive testing and documentation"),)),),
        (read, bash),
        GenParams(max_tokens=512),
    )
    state = current_action_state(conv, Settings())
    assert state.name == "inspect"
