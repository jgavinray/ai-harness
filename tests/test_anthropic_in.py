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
