import json

import httpx

from harness.config import Settings
from harness.server import create_app
from tests.fake_openai import FakeOpenAI, finish_chunk, text_chunk, tool_chunk

READ_TOOL = {
    "name": "Read",
    "description": "Reads a file",
    "input_schema": {
        "type": "object",
        "properties": {"file_path": {"type": "string"}},
        "required": ["file_path"],
    },
}


def request_body(stream: bool = True, system=None, tools=None) -> dict:
    return {
        "model": "claude-sonnet-4-6",
        "max_tokens": 512,
        "stream": stream,
        "system": system or "be brief",
        "messages": [{"role": "user", "content": "read /x"}],
        "tools": tools if tools is not None else [READ_TOOL],
    }


def make_client(fake: FakeOpenAI) -> httpx.AsyncClient:
    settings = Settings()
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    settings.backend.base_url = "http://fake/v1"
    app = create_app(settings, backend_client=backend_client)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")


def sse_events(raw: str):
    out = []
    for chunk in raw.strip().split("\n\n"):
        lines = chunk.split("\n")
        out.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return out


async def test_streaming_round_trip():
    fake = FakeOpenAI()
    fake.push([
        text_chunk("on it"),
        tool_chunk("c1", "Read", '{"file_path": "/x"}'),
        finish_chunk("tool_calls"),
    ])
    async with make_client(fake) as client:
        resp = await client.post("/v1/messages", json=request_body())
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    evs = sse_events(resp.text)
    names = [n for n, _ in evs]
    assert names[0] == "message_start" and names[-1] == "message_stop"
    tool_start = next(
        d for n, d in evs
        if n == "content_block_start" and d["content_block"]["type"] == "tool_use"
    )
    assert tool_start["content_block"]["name"] == "Read"
    md = next(d for n, d in evs if n == "message_delta")
    assert md["delta"]["stop_reason"] == "tool_use"


async def test_history_budget_uses_routed_backend_window():
    # Fleet mode: the compaction budget must come from the routed backend's
    # context_window, not the global single-backend profile default.
    from harness.config import PoolBackendCfg

    fake = FakeOpenAI()
    fake.push([text_chunk("ok"), finish_chunk("stop")])
    settings = Settings()
    settings.backends = [
        PoolBackendCfg(
            name="big",
            base_url="http://fake/v1",
            model="m",
            context_window=131072,
            roles=["main", "subagent", "fast"],
        )
    ]
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")

    body = request_body(stream=False)
    body["max_tokens"] = 64000
    # ~25k tokens of history: over budget if the 32768 default window is
    # used, comfortably under budget for the configured 131072 window.
    filler = "x " * 25000
    body["messages"] = [
        {"role": "user", "content": "the real task: explain fireshield"},
        {"role": "assistant", "content": filler},
        {"role": "user", "content": "go on"},
        {"role": "assistant", "content": "step two"},
        {"role": "user", "content": "go on"},
        {"role": "assistant", "content": "step three"},
        {"role": "user", "content": "finish up"},
    ]
    async with client:
        resp = await client.post("/v1/messages", json=body)
    assert resp.status_code == 200
    sent = json.dumps(fake.requests[-1])
    assert "the real task: explain fireshield" in sent
    assert "elided by harness" not in sent


async def test_non_streaming():
    fake = FakeOpenAI()
    fake.push([text_chunk("done"), finish_chunk("stop")])
    async with make_client(fake) as client:
        resp = await client.post("/v1/messages", json=request_body(stream=False))
    body = resp.json()
    assert body["type"] == "message"
    assert body["content"][0] == {"type": "text", "text": "done"}
    assert body["stop_reason"] == "end_turn"
    # usage must survive even though the client didn't stream
    assert body["usage"]["input_tokens"] == 10
    assert body["usage"]["output_tokens"] == 5


async def test_count_tokens_no_backend_call():
    fake = FakeOpenAI()
    async with make_client(fake) as client:
        resp = await client.post(
            "/v1/messages/count_tokens",
            json={"model": "m", "messages": [{"role": "user", "content": "hello world"}]},
        )
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] > 0
    assert fake.requests == []


async def test_backend_down_maps_to_overloaded():
    settings = Settings()
    settings.backend.base_url = "http://localhost:1/v1"  # nothing listens
    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        resp = await client.post("/v1/messages", json=request_body(stream=False))
    assert resp.status_code == 529
    assert resp.json()["error"]["type"] == "overloaded_error"


async def test_backend_500_streaming_emits_error_event():
    fake = FakeOpenAI()
    fake.push([{"_status": 500}])
    async with make_client(fake) as client:
        resp = await client.post("/v1/messages", json=request_body())
    assert resp.status_code == 200  # stream already started; error travels in-band
    evs = sse_events(resp.text)
    assert evs[-1][0] == "error"
    assert evs[-1][1]["error"]["type"] == "overloaded_error"


async def test_malformed_request_400():
    fake = FakeOpenAI()
    async with make_client(fake) as client:
        resp = await client.post("/v1/messages", json={"model": "m"})  # no messages/max_tokens
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


async def test_stats():
    fake = FakeOpenAI()
    fake.push([text_chunk("hi"), finish_chunk("stop")])
    async with make_client(fake) as client:
        await client.post("/v1/messages", json=request_body(stream=False))
        resp = await client.get("/stats")
    assert resp.json()["requests"] == 1


async def test_pipeline_applied_end_to_end():
    fake = FakeOpenAI()
    fake.push([text_chunk("ok"), finish_chunk("stop")])
    cc_system = (
        "You are Claude Code, Anthropic's official CLI for Claude.\n\n"
        "# Tone and style\n" + ("Be concise. " * 500) + "\n\n"
        "# Environment\nWorking directory: /repo\n"
    )
    many_tools = [READ_TOOL] + [
        {**READ_TOOL, "name": f"Extra{i}", "description": "x"} for i in range(14)
    ]
    async with make_client(fake) as client:
        await client.post(
            "/v1/messages", json=request_body(stream=False, system=cc_system, tools=many_tools)
        )
    sent = fake.requests[0]
    assert sent["messages"][0]["role"] == "system"
    assert len(sent["messages"][0]["content"]) < 5000
    assert "Working directory: /repo" in sent["messages"][0]["content"]
    assert len(sent["tools"]) <= 8


def _fleet_toml(roles: str) -> str:
    return (
        '[[backends]]\nname = "alpha"\nbase_url = "http://fake/v1"\n'
        f'model = "m"\ncontext_window = 32768\nroles = [{roles}]\n'
    )


async def test_admin_reload_applies_new_roles_and_keeps_stats(tmp_path):
    from harness.config import load_settings

    cfg = tmp_path / "harness.toml"
    cfg.write_text(_fleet_toml('"main"'))
    settings = load_settings(cfg)

    fake = FakeOpenAI()
    fake.push([text_chunk("ok"), finish_chunk("stop")])
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client, config_path=cfg)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")

    async with client:
        resp = await client.post("/v1/messages", json=request_body(stream=False))
        assert resp.status_code == 200

        cfg.write_text(_fleet_toml('"main", "fast"'))
        resp = await client.post("/admin/reload")
        assert resp.status_code == 200
        body = resp.json()
        assert body["updated"] == ["alpha"]

        stats = (await client.get("/stats")).json()
    assert stats["backends"]["alpha"]["roles"] == ["main", "fast"]
    assert stats["backends"]["alpha"]["requests"] == 1  # counter survived the reload
    assert stats["requests"] == 1


async def test_admin_reload_without_config_path_is_400():
    fake = FakeOpenAI()
    async with make_client(fake) as client:
        resp = await client.post("/admin/reload")
    assert resp.status_code == 400


async def test_stats_rehydrated_from_request_log(tmp_path):
    log = tmp_path / "requests.jsonl"
    records = [
        {"backend": "default", "input_tokens": 100, "output_tokens": 10,
         "cached_tokens": 40, "ttft_ms": 120},
        # response-cache hit: counted in tokens but not backend.requests
        {"backend": "default", "cache": "response", "input_tokens": 100,
         "output_tokens": 10, "cached_tokens": 0, "ttft_ms": 5},
        {"backend": "default", "input_tokens": 0, "output_tokens": 0,
         "cached_tokens": 0, "error": "boom"},
        # backend no longer in the fleet: global totals only
        {"backend": "ghost", "input_tokens": 50, "output_tokens": 5,
         "cached_tokens": 0, "ttft_ms": 80},
    ]
    log.write_text("\n".join(json.dumps(r) for r in records) + "\nnot json\n")

    settings = Settings()
    settings.backend.base_url = "http://fake/v1"
    settings.log.requests_path = str(log)
    fake = FakeOpenAI()
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    async with client:
        stats = (await client.get("/stats")).json()

    assert stats["requests"] == 4
    assert stats["errors"] == 1
    assert stats["input_tokens"] == 250
    assert stats["output_tokens"] == 25
    assert stats["cached_tokens"] == 40
    d = stats["backends"]["default"]
    assert d["requests"] == 2  # cache hit excluded, ghost unknown
    assert d["errors"] == 1
    assert d["kv_cache_hit_pct"] == 20.0  # 40 cached / 200 prompt
    assert d["kv_written_tokens"] == 180  # (200-40) prefill + 20 decode
    assert d["kv_cache_hit_pct_recent"] == 20.0  # window == full history here
    assert d["kv_used_pct"] is None  # openai-kind backend exposes no gauge


async def test_recent_cache_hit_window_reflects_current_behavior(tmp_path):
    # 5 old perfect-hit records pushed out of the recent window by 100 misses:
    # lifetime pct stays diluted, recent pct tells the truth about now.
    log = tmp_path / "requests.jsonl"
    old = [{"backend": "default", "input_tokens": 100, "output_tokens": 0,
            "cached_tokens": 100, "ttft_ms": 1}] * 5
    new = [{"backend": "default", "input_tokens": 100, "output_tokens": 0,
            "cached_tokens": 0, "ttft_ms": 1}] * 100
    log.write_text("\n".join(json.dumps(r) for r in old + new) + "\n")
    settings = Settings()
    settings.backend.base_url = "http://fake/v1"
    settings.log.requests_path = str(log)
    fake = FakeOpenAI()
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    async with client:
        d = (await client.get("/stats")).json()["backends"]["default"]
    assert d["kv_cache_hit_pct"] == 4.8  # 500 / 10500 lifetime
    assert d["kv_cache_hit_pct_recent"] == 0.0  # last 100 requests


async def test_stats_polls_live_kv_usage_from_backend_metrics():
    from harness.config import PoolBackendCfg

    fake = FakeOpenAI()
    fake.metrics_text = 'vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.42\n'
    settings = Settings()
    settings.backends = [
        PoolBackendCfg(name="v", kind="vllm", base_url="http://fake/v1",
                       model="m", roles=["main", "subagent", "fast"]),
    ]
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    async with client:
        d = (await client.get("/stats")).json()["backends"]["v"]
    assert d["kv_used_pct"] == 42.0


async def test_llamacpp_kv_used_estimated_from_slots_and_sessions(tmp_path):
    # llama.cpp dropped its KV gauges; estimate residency from slot capacity
    # and the last request size of the sessions most recently on this backend.
    from harness.config import PoolBackendCfg

    log = tmp_path / "requests.jsonl"
    records = [
        {"backend": "g", "session_key": "sA", "input_tokens": 300,
         "output_tokens": 50, "cached_tokens": 0, "ttft_ms": 1},
        {"backend": "g", "session_key": "sB", "input_tokens": 100,
         "output_tokens": 10, "cached_tokens": 0, "ttft_ms": 1},
        # sA's later turn supersedes its earlier residency
        {"backend": "g", "session_key": "sA", "input_tokens": 400,
         "output_tokens": 0, "cached_tokens": 0, "ttft_ms": 1},
    ]
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    fake = FakeOpenAI()
    fake.slots = [{"id": 0, "n_ctx": 1000}, {"id": 1, "n_ctx": 1000}]
    settings = Settings()
    settings.log.requests_path = str(log)
    settings.backends = [
        PoolBackendCfg(name="g", kind="llamacpp", base_url="http://fake/v1",
                       model="m", roles=["main", "subagent", "fast"]),
    ]
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    async with client:
        d = (await client.get("/stats")).json()["backends"]["g"]
    # resident = sA 400 + sB 110 = 510 of 2000 cells
    assert d["kv_used_pct"] == 25.5
    assert d["kv_used_est"] is True


async def test_vllm_kv_used_is_measured_not_estimated():
    from harness.config import PoolBackendCfg

    fake = FakeOpenAI()
    fake.metrics_text = 'vllm:kv_cache_usage_perc{engine="0"} 0.42\n'
    settings = Settings()
    settings.backends = [
        PoolBackendCfg(name="v", kind="vllm", base_url="http://fake/v1",
                       model="m", roles=["main", "subagent", "fast"]),
    ]
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    async with client:
        d = (await client.get("/stats")).json()["backends"]["v"]
    assert d["kv_used_pct"] == 42.0
    assert d["kv_used_est"] is False
