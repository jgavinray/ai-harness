import httpx

from harness.cache import ResponseCache, payload_key
from harness.config import Settings
from harness.ir import Done, TextDelta
from harness.server import create_app
from tests.fake_openai import FakeOpenAI, finish_chunk, text_chunk
from tests.test_server import request_body


def test_payload_key_ignores_stream_flags():
    a = {"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": True,
         "stream_options": {"include_usage": True}}
    b = {**a, "stream": False}
    del b["stream_options"]
    assert payload_key(a) == payload_key(b)
    c = {**a, "messages": [{"role": "user", "content": "y"}]}
    assert payload_key(c) != payload_key(a)


def test_cache_ttl_and_lru():
    cache = ResponseCache(ttl_s=0.0, max_entries=2)  # everything expires instantly
    cache.put("k", [TextDelta("v")])
    assert cache.get("k") is None  # expired
    cache = ResponseCache(ttl_s=60, max_entries=2)
    cache.put("a", [TextDelta("1")])
    cache.put("b", [TextDelta("2")])
    cache.put("c", [TextDelta("3")])  # evicts "a"
    assert cache.get("a") is None and cache.get("b") is not None


def haiku_body() -> dict:
    body = request_body(stream=False, tools=[])
    body["model"] = "claude-haiku-4-5"
    return body


def make_client(fake: FakeOpenAI, settings: Settings | None = None) -> httpx.AsyncClient:
    settings = settings or Settings()
    settings.backend.base_url = "http://fake/v1"
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")


async def test_identical_haiku_requests_hit_cache():
    fake = FakeOpenAI()
    fake.push([text_chunk("title: foo"), finish_chunk("stop")])
    async with make_client(fake) as client:
        r1 = await client.post("/v1/messages", json=haiku_body())
        r2 = await client.post("/v1/messages", json=haiku_body())
        stats = (await client.get("/stats")).json()
    assert r1.json()["content"] == r2.json()["content"]
    assert len(fake.requests) == 1  # second served from cache
    assert stats["response_cache"]["hits"] == 1


async def test_main_role_not_cached():
    fake = FakeOpenAI()
    fake.push([text_chunk("work"), finish_chunk("stop")])
    async with make_client(fake) as client:
        await client.post("/v1/messages", json=request_body(stream=False, tools=[]))
        await client.post("/v1/messages", json=request_body(stream=False, tools=[]))
    assert len(fake.requests) == 2


async def test_stats_per_backend_shape():
    fake = FakeOpenAI()
    fake.push([text_chunk("x"), finish_chunk("stop")])
    async with make_client(fake) as client:
        await client.post("/v1/messages", json=request_body(stream=False, tools=[]))
        stats = (await client.get("/stats")).json()
    b = stats["backends"]["default"]
    assert b["requests"] == 1
    assert "ttft_p50_ms" in b and "kv_cache_hit_pct" in b
    assert b["roles"] == ["main", "subagent", "fast"]


async def test_dashboard_served():
    fake = FakeOpenAI()
    async with make_client(fake) as client:
        resp = await client.get("/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "ai-harness" in resp.text
