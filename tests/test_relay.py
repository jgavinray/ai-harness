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
