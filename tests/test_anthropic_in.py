import json
from pathlib import Path

from harness.codec.anthropic_in import decode
from harness.ir import TextPart, ToolCallPart, ToolResultPart


def fixture():
    return json.loads(Path("tests/fixtures/cc_request.json").read_text())


def test_decode_basics():
    conv = decode(fixture())
    assert "You are Claude Code" in conv.system
    assert "# Environment" in conv.system  # blocks joined
    assert conv.params.stream is True
    assert conv.params.max_tokens == 8192
    assert len(conv.tools) == 2
    assert conv.tools[0].original_schema == conv.tools[0].input_schema
    assert conv.tools[0].name == "Read"


def test_decode_tool_roundtrip():
    conv = decode(fixture())
    calls = [p for t in conv.turns for p in t.parts if isinstance(p, ToolCallPart)]
    results = [p for t in conv.turns for p in t.parts if isinstance(p, ToolResultPart)]
    assert len(calls) == 2 and len(results) == 2
    assert calls[0].id == results[0].tool_call_id == "toolu_01A"
    assert "def test_add" in results[0].content  # block-list content flattened
    assert results[1].is_error is True


def test_string_content_becomes_text_part():
    conv = decode(fixture())
    assert conv.turns[0].parts == (TextPart("fix the failing test in tests/test_utils.py"),)


def test_unsupported_blocks_replaced():
    conv = decode(fixture())
    last_parts = conv.turns[-1].parts
    assert any(isinstance(p, TextPart) and "unsupported" in p.text for p in last_parts)


def test_string_system_supported():
    body = fixture()
    body["system"] = "plain system"
    assert decode(body).system == "plain system"


def test_missing_optional_fields():
    conv = decode({"model": "m", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]})
    assert conv.system == ""
    assert conv.tools == ()
    assert conv.params.stream is False
    assert conv.params.response_schema is None


# Claude Code runs WebSearch as its own Messages API call carrying the
# server-side tool definition below. It has no "input_schema" — the API, not
# the model, executes it — and indexing that key returned HTTP 400
# "could not decode request: KeyError('input_schema')" for the whole request.
WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 8,
}


def test_server_side_tool_definition_is_dropped_not_fatal():
    body = fixture()
    body["tools"] = [WEB_SEARCH_TOOL, *body["tools"]]
    conv = decode(body)
    assert [t.name for t in conv.tools] == ["Read", "Bash"]


def test_server_side_tool_only_request_decodes_to_no_tools():
    body = fixture()
    body["tools"] = [WEB_SEARCH_TOOL]
    assert decode(body).tools == ()


# Claude Code 2.1.220 asks for structured replies with output_config.format;
# ignoring it let the backend answer in prose, which the client cannot parse.
TITLE_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}},
    "required": ["title"],
    "additionalProperties": False,
}


def test_output_config_json_schema_becomes_response_schema():
    body = fixture()
    body["output_config"] = {
        "effort": "high",
        "format": {"type": "json_schema", "schema": TITLE_SCHEMA},
    }
    assert decode(body).params.response_schema == TITLE_SCHEMA


def test_output_config_without_format_has_no_response_schema():
    body = fixture()
    body["output_config"] = {"effort": "xhigh"}
    assert decode(body).params.response_schema is None


def test_output_config_unknown_format_type_is_ignored():
    body = fixture()
    body["output_config"] = {"format": {"type": "text"}}
    assert decode(body).params.response_schema is None
