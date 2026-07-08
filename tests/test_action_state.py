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


def test_question_brief_mentioning_read_does_not_require_tool():
    # eval regression (envelope-27b, find-and-report 10/20): the task prompt
    # contains "read" so inspect state required a tool every turn, making the
    # demanded prose answer illegal — the give-up contract then replaced the
    # model's correct answer with an honest failure.
    prompt = (
        "Where does this project read the WORKER_POOL_SIZE environment "
        "variable? Reply with the file path and the line number in the "
        "format path:line. Do not modify any files."
    )
    conv = Conversation(
        "sys",
        (Turn("user", (TextPart(prompt),)),),
        _tools(),
        GenParams(max_tokens=512),
    )
    state = current_action_state(conv, Settings())
    assert state.name == "inspect"
    assert state.requires_tool is False


def test_short_inspect_instruction_still_requires_tool():
    conv = Conversation(
        "sys",
        (Turn("user", (TextPart("read x"),)),),
        _tools(),
        GenParams(max_tokens=512),
    )
    state = current_action_state(conv, Settings())
    assert state.name == "inspect"
    assert state.requires_tool is True


def _edit_turns(n: int):
    turns = [Turn("user", (TextPart("rename calc_total to compute_total everywhere in this project"),)),
             Turn("assistant", (ToolCallPart("r1", "Read", {"file_path": "/x/a.py"}),)),
             Turn("user", (ToolResultPart("r1", "code"),))]
    for i in range(n):
        turns.append(Turn("assistant", (ToolCallPart(f"e{i}", "Edit", {
            "file_path": "/x/a.py", "old_string": "a", "new_string": "b"}),)))
        turns.append(Turn("user", (ToolResultPart(f"e{i}", "edited"),)))
    return tuple(turns)


def test_change_set_edits_allowed_below_unverified_limit():
    # eval regression (envelope-27b-rename-fix2, 10/20): verify state bound
    # after every single edit, so multi-spot renames could never finish —
    # the third Edit of the change-set was denied.
    conv = Conversation("sys", _edit_turns(2), _tools(), GenParams(max_tokens=512))
    state = current_action_state(conv, Settings())
    assert state.name == "edit_existing"
    assert "Edit" in state.allowed_tools
    assert "Bash" in state.allowed_tools  # voluntary mid-change-set verification


def test_verify_binds_at_unverified_edit_limit():
    settings = Settings()
    settings.pipeline.unverified_edit_limit = 3
    conv = Conversation("sys", _edit_turns(3), _tools(), GenParams(max_tokens=512))
    state = current_action_state(conv, settings)
    assert state.name == "verify"
    assert "Edit" not in state.allowed_tools


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
