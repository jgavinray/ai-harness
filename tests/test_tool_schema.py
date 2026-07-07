from harness.config import Settings
from harness.ir import Conversation, GenParams, ToolDef
from harness.pipeline.tool_schema import ToolSchemaStage, trim

LONG_DESC = ("This tool reads files. " + "It supports many advanced options. " * 40).strip()

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "title": "ReadInput",
    "additionalProperties": False,
    "propertyNames": {"pattern": "^[a-z_]+$"},
    "properties": {
        "file_path": {
            "type": "string",
            "description": "The absolute path. " + "More words about paths. " * 20,
        },
        "limit": {"anyOf": [{"type": "number"}, {"type": "null"}]},
    },
    "required": ["file_path"],
}


def make_conv() -> Conversation:
    return Conversation(
        "s", (), (ToolDef("Read", LONG_DESC, SCHEMA, SCHEMA),), GenParams(max_tokens=10)
    )


def test_simplification():
    out = ToolSchemaStage().apply(make_conv(), Settings())
    t = out.tools[0]
    assert len(t.description) <= 300
    assert t.description.endswith(".")
    s = t.input_schema
    assert "$schema" not in s and "title" not in s and "additionalProperties" not in s
    assert "propertyNames" not in s
    assert len(s["properties"]["file_path"]["description"]) <= 150
    # anyOf [X, null] flattened to X
    assert s["properties"]["limit"] == {"type": "number"}
    # original untouched
    assert t.original_schema is SCHEMA


def test_all_tools_are_simplified_for_hidden_tool_surfacing():
    conv = make_conv()
    conv = Conversation(conv.system, conv.turns, (), conv.params, conv.tools)
    out = ToolSchemaStage().apply(conv, Settings())
    assert out.tools == ()
    assert "propertyNames" not in out.all_tools[0].input_schema
    assert out.all_tools[0].original_schema is SCHEMA


def test_trim_sentence_boundary():
    assert trim("One. Two. Three.", 10) == "One. Two."
    assert trim("short", 10) == "short"
    assert trim("noboundaryatallinthisstring" * 5, 20).endswith("…")
