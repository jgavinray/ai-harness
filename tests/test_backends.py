import httpx
import pytest

from harness.backends.base import BackendError
from harness.backends.openai_compat import (
    LlamaCppBackend,
    OpenAIBackend,
    VllmBackend,
    make_backend,
)
from harness.config import BackendCfg
from tests.fake_openai import FakeOpenAI, finish_chunk, text_chunk


def make(fake: FakeOpenAI, kind: str = "openai"):
    cfg = BackendCfg(kind=kind, base_url="http://fake/v1")
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=fake.app), base_url="http://fake")
    return make_backend(cfg, client)


async def test_stream_yields_chunks():
    fake = FakeOpenAI()
    fake.push([text_chunk("hi"), finish_chunk()])
    chunks = [c async for c in make(fake).stream({"model": "m", "messages": []})]
    assert chunks[0]["choices"][0]["delta"]["content"] == "hi"
    assert chunks[-1]["usage"]["prompt_tokens"] == 10
    assert fake.requests[0]["model"] == "m"


async def test_http_error_raises():
    fake = FakeOpenAI()
    fake.push([{"_status": 500}])
    with pytest.raises(BackendError):
        [c async for c in make(fake).stream({"model": "m", "messages": []})]


async def test_midstream_death_raises():
    fake = FakeOpenAI()
    fake.push([text_chunk("partial"), {"_die_midstream": True}])
    with pytest.raises(BackendError):
        async for _ in make(fake).stream({"model": "m", "messages": []}):
            pass


def test_constraints():
    schema = {"type": "object"}
    p = {"model": "m"}
    assert OpenAIBackend.constrained is False
    assert VllmBackend.constrained and LlamaCppBackend.constrained
    assert OpenAIBackend.apply_constraint(None, dict(p), schema) == p
    v = VllmBackend.apply_constraint(None, dict(p), schema)
    assert v["guided_json"] == schema and v["tool_choice"] == "required"
    l = LlamaCppBackend.apply_constraint(None, dict(p), schema)
    assert l["json_schema"] == schema


def test_factory():
    fake = FakeOpenAI()
    assert isinstance(make(fake, "vllm"), VllmBackend)
    assert isinstance(make(fake, "llamacpp"), LlamaCppBackend)
    assert isinstance(make(fake, "openai"), OpenAIBackend)
    with pytest.raises(ValueError):
        make(fake, "wat")
