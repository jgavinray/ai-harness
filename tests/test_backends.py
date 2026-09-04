import httpx
import pytest

from harness.backends.base import BackendError
from harness.backends.openai_compat import (
    LlamaCppBackend,
    OpenAIBackend,
    SglangBackend,
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
    chunks = [c async for c in make(fake).stream({"model": "m", "messages": [], "stream": True})]
    assert chunks[0]["choices"][0]["delta"]["content"] == "hi"
    assert chunks[-1]["usage"]["prompt_tokens"] == 10
    assert fake.requests[0]["model"] == "m"


async def test_http_error_raises():
    fake = FakeOpenAI()
    fake.push([{"_status": 500}])
    with pytest.raises(BackendError):
        [c async for c in make(fake).stream({"model": "m", "messages": [], "stream": True})]


async def test_midstream_death_raises():
    fake = FakeOpenAI()
    fake.push([text_chunk("partial"), {"_die_midstream": True}])
    with pytest.raises(BackendError):
        async for _ in make(fake).stream({"model": "m", "messages": [], "stream": True}):
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
    assert SglangBackend.constrained
    sg = SglangBackend.apply_constraint(SglangBackend(None, None), dict(p), schema)
    assert sg["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": schema, "strict": True},
    }
    assert sg["tool_choice"] == "required"


def test_vllm_constraint_schema_strips_unsupported_grammar_keys():
    schema = {
        "type": "object",
        "propertyNames": {"pattern": "^[a-z]+$"},
        "additionalProperties": False,
        "properties": {"path": {"type": "string", "description": "drop", "pattern": "^/"}},
        "required": ["path"],
    }
    out = VllmBackend.apply_constraint(None, {"model": "m"}, schema)["guided_json"]
    assert out == {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }


def test_factory():
    fake = FakeOpenAI()
    assert isinstance(make(fake, "vllm"), VllmBackend)
    assert isinstance(make(fake, "sglang"), SglangBackend)
    assert isinstance(make(fake, "llamacpp"), LlamaCppBackend)
    assert isinstance(make(fake, "openai"), OpenAIBackend)
    with pytest.raises(ValueError):
        make(fake, "wat")
