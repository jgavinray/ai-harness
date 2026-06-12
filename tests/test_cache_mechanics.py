import json

import httpx

from harness.codec.anthropic_out import collect, stream_sse
from harness.config import BackendCfg, Settings
from harness.ir import Done, TextDelta
from harness.profiles.registry import get_profile
from tests.fake_openai import FakeOpenAI, text_chunk


async def aiter(items):
    for it in items:
        yield it


async def test_parse_vllm_cached_tokens():
    chunks = [
        text_chunk("hi"),
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 800},
            },
        },
    ]
    evs = [e async for e in get_profile("qwen").parse(aiter(chunks))]
    done = evs[-1]
    assert done.input_tokens == 1000 and done.cached_tokens == 800


async def test_parse_llamacpp_timings_cached():
    # llama.cpp reports tokens actually evaluated in timings.prompt_n;
    # cached = prompt_tokens - prompt_n
    chunks = [
        text_chunk("hi"),
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 20},
            "timings": {"prompt_n": 150},
        },
    ]
    evs = [e async for e in get_profile("gemma").parse(aiter(chunks))]
    done = evs[-1]
    assert done.cached_tokens == 850


async def test_sse_usage_reports_cache_read():
    async def events():
        yield TextDelta("x")
        yield Done("end_turn", input_tokens=1000, output_tokens=5, cached_tokens=800)

    raw = "".join([s async for s in stream_sse(events(), "m", "msg")])
    md = next(
        json.loads(c.split("\n")[1].removeprefix("data: "))
        for c in raw.split("\n\n")
        if c.startswith("event: message_delta")
    )
    assert md["usage"]["cache_read_input_tokens"] == 800
    assert md["usage"]["input_tokens"] == 200  # uncached portion, Anthropic semantics


async def test_collect_usage_cache_read():
    msg = collect([TextDelta("x"), Done("end_turn", 1000, 5, cached_tokens=800)], "m", "id")
    assert msg["usage"] == {
        "input_tokens": 200,
        "output_tokens": 5,
        "cache_read_input_tokens": 800,
        "cache_creation_input_tokens": 0,
    }


async def test_llamacpp_sends_cache_prompt():
    from harness.backends.openai_compat import LlamaCppBackend
    from tests.fake_openai import FakeOpenAI, finish_chunk

    fake = FakeOpenAI()
    fake.push([finish_chunk("stop")])
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=fake.app), base_url="http://fake")
    backend = LlamaCppBackend(BackendCfg(base_url="http://fake/v1"), client)
    [c async for c in backend.stream({"model": "m", "messages": []})]
    assert fake.requests[0]["cache_prompt"] is True


def test_history_hysteresis():
    from harness.pipeline.history import HistoryStage, TARGET_RATIO
    from harness.tokens.counter import count_conversation
    from tests.test_history import big_session, small_settings

    # window chosen so the protected tail is well under the compaction
    # target; otherwise the target is unreachable and the assertion is vacuous
    s = small_settings(16000)
    stage = HistoryStage()
    out = stage.apply(big_session(40, 8000), s)
    budget = s.profile.context_window - 1024 - 1024  # max_tokens=1024, margin
    # compacted well below budget so next turn doesn't immediately re-evict
    assert count_conversation(out, stage.counter) <= budget * TARGET_RATIO
