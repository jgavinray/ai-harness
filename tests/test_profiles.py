import json

import pytest

from harness.ir import (
    Conversation,
    Done,
    GenParams,
    TextDelta,
    TextPart,
    ThinkingDelta,
    ToolCall,
    ToolCallPart,
    ToolDef,
    ToolResultPart,
    Turn,
)
from harness.profiles.base import TagSplitter
from harness.profiles.registry import get_profile


def conv() -> Conversation:
    return Conversation(
        system="be helpful",
        turns=(
            Turn("user", (TextPart("fix it"),)),
            Turn(
                "assistant",
                (TextPart("looking"), ToolCallPart("t1", "Read", {"file_path": "/x"})),
            ),
            Turn("user", (ToolResultPart("t1", "contents"),)),
        ),
        tools=(ToolDef("Read", "Reads a file", {"type": "object", "properties": {}}, {"type": "object"}),),
        params=GenParams(max_tokens=2048, temperature=0.2, stop_sequences=("STOP",), stream=True),
    )


def chunk(delta: dict, finish: str | None = None, usage: dict | None = None) -> dict:
    c: dict = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        c = {"choices": [], "usage": usage} if delta is None else c
        c["usage"] = usage
    return c


async def aiter(items):
    for it in items:
        yield it


# ---------- render ----------

def test_base_render():
    payload = get_profile("qwen").render(conv(), "qwen2.5-coder:14b")
    msgs = payload["messages"]
    assert msgs[0] == {"role": "system", "content": "be helpful"}
    assert msgs[1] == {"role": "user", "content": "fix it"}
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["content"] == "looking"
    tc = msgs[2]["tool_calls"][0]
    assert tc["id"] == "t1" and tc["type"] == "function"
    assert tc["function"]["name"] == "Read"
    assert json.loads(tc["function"]["arguments"]) == {"file_path": "/x"}
    assert msgs[3] == {"role": "tool", "tool_call_id": "t1", "content": "contents"}
    assert payload["tools"][0]["function"]["name"] == "Read"
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["max_tokens"] == 2048
    assert payload["stop"] == ["STOP"]
    assert payload["model"] == "qwen2.5-coder:14b"


def test_gemma_render_no_system_role():
    payload = get_profile("gemma").render(conv(), "gemma3:27b")
    msgs = payload["messages"]
    assert all(m["role"] != "system" for m in msgs)
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"].startswith("be helpful")
    assert "fix it" in msgs[0]["content"]


# ---------- parse ----------

async def test_base_parse_text_and_tool():
    chunks = [
        chunk({"content": "Hel"}),
        chunk({"content": "lo"}),
        chunk({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "Read", "arguments": '{"file_'}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'path": "/x"}'}}]}),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
         "usage": {"prompt_tokens": 11, "completion_tokens": 7}},
    ]
    evs = [e async for e in get_profile("qwen").parse(aiter(chunks))]
    assert evs[0] == TextDelta("Hel") and evs[1] == TextDelta("lo")
    call = next(e for e in evs if isinstance(e, ToolCall))
    assert call.name == "Read" and call.arguments == {"file_path": "/x"}
    done = evs[-1]
    assert done == Done("tool_use", 11, 7)


async def test_parse_malformed_args_kept_raw():
    chunks = [
        chunk({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "Read", "arguments": '{"file_path": '}}]}),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]
    evs = [e async for e in get_profile("qwen").parse(aiter(chunks))]
    call = next(e for e in evs if isinstance(e, ToolCall))
    assert call.arguments == {}
    assert call.raw_arguments == '{"file_path": '


async def test_parse_reasoning_content_field():
    chunks = [
        chunk({"reasoning_content": "thinking..."}),
        chunk({"content": "answer"}),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    evs = [e async for e in get_profile("qwen").parse(aiter(chunks))]
    assert evs[0] == ThinkingDelta("thinking...")
    assert evs[1] == TextDelta("answer")
    assert evs[-1] == Done("end_turn", 0, 0)


async def test_r1_think_tags_split_across_chunks():
    chunks = [
        chunk({"content": "<thi"}),
        chunk({"content": "nk>deep"}),
        chunk({"content": " thought</th"}),
        chunk({"content": "ink>result"}),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    evs = [e async for e in get_profile("deepseek_r1").parse(aiter(chunks))]
    thinking = "".join(e.text for e in evs if isinstance(e, ThinkingDelta))
    text = "".join(e.text for e in evs if isinstance(e, TextDelta))
    assert thinking == "deep thought"
    assert text == "result"


async def test_length_finish_maps_to_max_tokens():
    chunks = [{"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": "length"}]}]
    evs = [e async for e in get_profile("qwen").parse(aiter(chunks))]
    assert evs[-1] == Done("max_tokens", 0, 0)


# ---------- splitter / registry ----------

def test_tag_splitter_flush():
    sp = TagSplitter("<think>", "</think>")
    out = []
    for piece in ("plain <t", "hink>in", "side</think> after", ""):
        out.extend(sp.feed(piece))
    out.extend(sp.flush())
    text = "".join(t for kind, t in out if kind == "text")
    think = "".join(t for kind, t in out if kind == "think")
    assert text == "plain  after"
    assert think == "inside"


def test_registry():
    for name in ("qwen", "deepseek_r1", "devstral", "gemma"):
        assert get_profile(name).name == name
    with pytest.raises(ValueError):
        get_profile("gpt99")
