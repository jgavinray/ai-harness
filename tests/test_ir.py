import pytest

from harness.ir import (
    Conversation,
    Done,
    GenParams,
    TextPart,
    ToolCall,
    ToolCallPart,
    ToolDef,
    ToolResultPart,
    Turn,
)


def test_conversation_construction():
    conv = Conversation(
        system="be good",
        turns=(
            Turn("user", (TextPart("hi"),)),
            Turn("assistant", (ToolCallPart("t1", "Read", {"file_path": "/x"}),)),
            Turn("user", (ToolResultPart("t1", "contents"),)),
        ),
        tools=(ToolDef("Read", "Reads a file", {"type": "object"}, {"type": "object"}),),
        params=GenParams(max_tokens=4096),
    )
    assert conv.turns[1].parts[0].name == "Read"
    assert conv.tools[0].original_schema == {"type": "object"}


def test_frozen():
    with pytest.raises(Exception):
        TextPart("a").text = "b"


def test_event_defaults():
    assert ToolCall("t1", "Read", {}).raw_arguments == ""
    assert Done("end_turn").output_tokens == 0
