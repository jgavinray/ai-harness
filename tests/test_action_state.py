from harness.action_state import current_action_state, shape_tools_for_state
from harness.config import Settings
from harness.ir import Conversation, GenParams, TextPart, ToolDef, Turn


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
