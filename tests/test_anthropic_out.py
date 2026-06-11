import json

from harness.codec.anthropic_out import collect, error_sse, stream_sse
from harness.ir import Done, TextDelta, ThinkingDelta, ToolCall


async def events():
    yield ThinkingDelta("hm")
    yield TextDelta("Hel")
    yield TextDelta("lo")
    yield ToolCall("toolu_x1", "Read", {"file_path": "/x"})
    yield Done("tool_use", input_tokens=10, output_tokens=5)


def parse_sse(raw: str):
    out = []
    for chunk in raw.strip().split("\n\n"):
        lines = chunk.split("\n")
        name = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        out.append((name, data))
    return out


async def test_stream_sequence():
    raw = "".join([s async for s in stream_sse(events(), "test-model", "msg_1")])
    evs = parse_sse(raw)
    names = [n for n, _ in evs]
    assert names[0] == "message_start"
    assert evs[0][1]["message"]["model"] == "test-model"
    # ordered: thinking block, text block, tool block
    starts = [d for n, d in evs if n == "content_block_start"]
    assert [s["content_block"]["type"] for s in starts] == ["thinking", "text", "tool_use"]
    assert starts[2]["content_block"]["name"] == "Read"
    assert starts[2]["content_block"]["id"] == "toolu_x1"
    # indexes increase
    assert [s["index"] for s in starts] == [0, 1, 2]
    deltas = [d for n, d in evs if n == "content_block_delta"]
    text = "".join(d["delta"]["text"] for d in deltas if d["delta"]["type"] == "text_delta")
    assert text == "Hello"
    tool_json = "".join(
        d["delta"]["partial_json"] for d in deltas if d["delta"]["type"] == "input_json_delta"
    )
    assert json.loads(tool_json) == {"file_path": "/x"}
    md = [d for n, d in evs if n == "message_delta"][0]
    assert md["delta"]["stop_reason"] == "tool_use"
    assert md["usage"]["output_tokens"] == 5
    assert names[-1] == "message_stop"
    # every started block is stopped
    assert names.count("content_block_start") == names.count("content_block_stop") == 3


async def test_collect_non_streaming():
    msg = collect([e async for e in events()], "test-model", "msg_2")
    assert msg["type"] == "message"
    assert msg["stop_reason"] == "tool_use"
    kinds = [b["type"] for b in msg["content"]]
    assert kinds == ["thinking", "text", "tool_use"]
    assert msg["content"][1]["text"] == "Hello"
    assert msg["content"][2]["input"] == {"file_path": "/x"}
    assert msg["usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }


def test_error_sse():
    evs = parse_sse(error_sse("overloaded_error", "backend down"))
    assert evs[0][0] == "error"
    assert evs[0][1]["error"]["type"] == "overloaded_error"


async def test_text_only_stream_ends_end_turn():
    async def evs():
        yield TextDelta("hi")
        yield Done("end_turn", 1, 1)

    raw = "".join([s async for s in stream_sse(evs(), "m", "msg_3")])
    parsed = parse_sse(raw)
    md = [d for n, d in parsed if n == "message_delta"][0]
    assert md["delta"]["stop_reason"] == "end_turn"
