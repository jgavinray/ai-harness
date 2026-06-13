import httpx

from harness.config import Settings
from harness.ir import (
    Conversation,
    Done,
    GenParams,
    TextDelta,
    TextPart,
    ToolCall,
    ToolDef,
    Turn,
)
from harness.profiles.registry import get_profile
from harness.relay import run
from tests.fake_openai import FakeOpenAI, finish_chunk, text_chunk, tool_chunk
from tests.test_backends import make

READ_SCHEMA = {
    "type": "object",
    "properties": {"file_path": {"type": "string"}},
    "required": ["file_path"],
}


def conv() -> Conversation:
    return Conversation(
        "sys",
        (Turn("user", (TextPart("read x"),)),),
        (ToolDef("Read", "reads", READ_SCHEMA, READ_SCHEMA),),
        GenParams(max_tokens=512, stream=True),
    )


async def collect_events(fake: FakeOpenAI, kind: str = "openai", settings: Settings | None = None):
    settings = settings or Settings()
    backend = make(fake, kind)
    return [e async for e in run(conv(), get_profile("qwen"), backend, settings)]


async def test_happy_path():
    fake = FakeOpenAI()
    fake.push([
        text_chunk("ok"),
        tool_chunk("c1", "Read", '{"file_path": "/x"}'),
        finish_chunk("tool_calls"),
    ])
    evs = await collect_events(fake)
    assert TextDelta("ok") in evs
    assert any(isinstance(e, ToolCall) and e.arguments == {"file_path": "/x"} for e in evs)
    assert evs[-1].stop_reason == "tool_use"
    assert len(fake.requests) == 1


async def test_bad_then_good_retries_with_feedback():
    fake = FakeOpenAI()
    fake.push([
        text_chunk("trying"),
        tool_chunk("c1", "Read", '{"wrong_param": 1}'),
        finish_chunk("tool_calls"),
    ])
    fake.push([
        text_chunk("retry noise"),
        tool_chunk("c2", "Read", '{"file_path": "/x"}'),
        finish_chunk("tool_calls"),
    ])
    fake.push([finish_chunk("stop")])  # safety
    evs = await collect_events(fake)
    assert len(fake.requests) == 2
    # feedback message present in the second request
    second_msgs = fake.requests[1]["messages"]
    assert any("file_path" in str(m.get("content")) and m["role"] == "user" for m in second_msgs)
    # retry text suppressed, valid call emitted
    assert TextDelta("retry noise") not in evs
    assert any(isinstance(e, ToolCall) and e.arguments == {"file_path": "/x"} for e in evs)


async def test_retries_exhausted_degrades_to_text():
    fake = FakeOpenAI()
    bad = [tool_chunk("c1", "Read", '{"nope": 1}'), finish_chunk("tool_calls")]
    fake.push(bad)
    fake.push(bad)
    fake.push(bad)  # repeats forever
    evs = await collect_events(fake)
    assert len(fake.requests) == 3  # initial + 2 retries
    assert not any(isinstance(e, ToolCall) for e in evs)
    assert any(isinstance(e, TextDelta) and "invalid tool call" in e.text for e in evs)
    assert evs[-1].stop_reason == "end_turn"


async def test_constrained_backend_gets_schema_on_retry():
    fake = FakeOpenAI()
    fake.push([tool_chunk("c1", "Read", '{"nope": 1}'), finish_chunk("tool_calls")])
    fake.push([tool_chunk("c2", "Read", '{"file_path": "/x"}'), finish_chunk("tool_calls")])
    fake.push([finish_chunk("stop")])
    await collect_events(fake, kind="vllm")
    assert "guided_json" not in fake.requests[0]
    assert fake.requests[1]["guided_json"] == READ_SCHEMA


async def test_degenerate_stream_aborted():
    fake = FakeOpenAI()
    fake.push([text_chunk("loop loop loop ")] * 300 + [finish_chunk("stop")])
    evs = await collect_events(fake)
    assert isinstance(evs[-1], Done)
    assert evs[-1].stop_reason == "end_turn"
    streamed = "".join(e.text for e in evs if isinstance(e, TextDelta))
    assert len(streamed) < 4500  # aborted well before 300 chunks


def conv_with_repeats(n: int) -> Conversation:
    from harness.ir import ToolCallPart, ToolResultPart
    turns: list[Turn] = [Turn("user", (TextPart("find the config"),))]
    for i in range(n):
        turns.append(Turn("assistant", (ToolCallPart(f"t{i}", "Read", {"file_path": "/x"}),)))
        turns.append(Turn("user", (ToolResultPart(f"t{i}", "same content"),)))
    return Conversation(
        "sys", tuple(turns),
        (ToolDef("Read", "reads", READ_SCHEMA, READ_SCHEMA),),
        GenParams(max_tokens=512, stream=True),
    )


async def test_cross_turn_loop_broken_with_feedback():
    import json
    # history already holds the identical call 3x; the 4th must trigger
    # loop-break feedback instead of being yielded
    fake = FakeOpenAI()
    fake.push([tool_chunk("c9", "Read", '{"file_path": "/x"}'), finish_chunk("tool_calls")])
    fake.push([text_chunk("the config is in /x; done"), finish_chunk("stop")])
    backend = make(fake, "openai")
    evs = [e async for e in run(conv_with_repeats(3), get_profile("qwen"), backend, Settings())]
    assert not any(isinstance(e, ToolCall) for e in evs)
    assert len(fake.requests) == 2
    assert "identical" in json.dumps(fake.requests[1])
    assert evs[-1].stop_reason == "end_turn"


async def test_two_prior_repeats_pass_through():
    # re-running a command a couple of times is legitimate (e.g. pytest
    # after a fix); only sustained repetition is broken
    fake = FakeOpenAI()
    fake.push([tool_chunk("c9", "Read", '{"file_path": "/x"}'), finish_chunk("tool_calls")])
    backend = make(fake, "openai")
    evs = [e async for e in run(conv_with_repeats(2), get_profile("qwen"), backend, Settings())]
    assert any(isinstance(e, ToolCall) for e in evs)
    assert len(fake.requests) == 1


WEB_SCHEMA = {
    "type": "object",
    "properties": {"url": {"type": "string"}},
    "required": ["url"],
}


def conv_with_hidden_tool() -> Conversation:
    read = ToolDef("Read", "reads", READ_SCHEMA, READ_SCHEMA)
    web = ToolDef("WebFetch", "fetches a url", WEB_SCHEMA, WEB_SCHEMA)
    return Conversation(
        "sys",
        (Turn("user", (TextPart("fetch x"),)),),
        (read,),                      # only Read is surfaced
        GenParams(max_tokens=512, stream=True),
        all_tools=(read, web),        # WebFetch is catalog-only
    )


async def test_hidden_tool_valid_call_passes_through():
    # Model called a catalogued-but-unsurfaced tool with valid args:
    # zero-cost path, no retry round-trip.
    fake = FakeOpenAI()
    fake.push([tool_chunk("c1", "WebFetch", '{"url": "https://x"}'),
               finish_chunk("tool_calls")])
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [e async for e in run(conv_with_hidden_tool(), get_profile("qwen"),
                                backend, Settings(), metrics=metrics)]
    assert any(isinstance(e, ToolCall) and e.name == "WebFetch" for e in evs)
    assert len(fake.requests) == 1
    assert metrics["tool_surfaced"] == 1


async def test_hidden_tool_invalid_call_swaps_schema_and_retries():
    fake = FakeOpenAI()
    fake.push([tool_chunk("c1", "WebFetch", '{"address": "x"}'),   # wrong param
               finish_chunk("tool_calls")])
    fake.push([tool_chunk("c2", "WebFetch", '{"url": "https://x"}'),
               finish_chunk("tool_calls")])
    fake.push([finish_chunk("stop")])  # safety
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [e async for e in run(conv_with_hidden_tool(), get_profile("qwen"),
                                backend, Settings(), metrics=metrics)]
    assert len(fake.requests) == 2
    # the retry request must offer the WebFetch schema
    retry_tools = [t["function"]["name"] for t in fake.requests[1].get("tools", [])]
    assert "WebFetch" in retry_tools
    # and the valid second call is emitted
    assert any(isinstance(e, ToolCall) and e.arguments == {"url": "https://x"} for e in evs)
    assert metrics["tool_surfaced"] == 1


async def test_truly_unknown_tool_still_fails_with_feedback():
    # A tool in neither the surfaced set nor the catalog keeps today's
    # behavior: feedback retry, then degrade to text.
    fake = FakeOpenAI()
    bad = [tool_chunk("c1", "Nonexistent", '{"a": 1}'), finish_chunk("tool_calls")]
    fake.push(bad)
    fake.push(bad)
    fake.push(bad)
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [e async for e in run(conv_with_hidden_tool(), get_profile("qwen"),
                                backend, Settings(), metrics=metrics)]
    assert not any(isinstance(e, ToolCall) for e in evs)
    assert metrics["tool_surfaced"] == 0
