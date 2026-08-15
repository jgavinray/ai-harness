import asyncio
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

EDIT_TOOL = {
    "name": "Edit",
    "description": "Edits a file",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
        },
        "required": ["file_path", "old_string", "new_string"],
    },
}
BASH_TOOL = {
    "name": "Bash",
    "description": "Runs a shell command",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}
GREP_TOOL = {
    "name": "Grep",
    "description": "Searches files",
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
        },
        "required": ["pattern"],
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


def make_client(fake: FakeOpenAI, path_aliases: list[list[str]] | None = None) -> httpx.AsyncClient:
    settings = Settings()
    if path_aliases:
        settings.pipeline.path_aliases = path_aliases
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
        resp = await client.post("/v1/messages", json=request_body(stream=False, tools=[]))
    body = resp.json()
    assert body["type"] == "message"
    assert body["content"][0] == {"type": "text", "text": "done"}
    assert body["stop_reason"] == "end_turn"
    # usage must survive even though the client didn't stream
    assert body["usage"]["input_tokens"] == 10
    assert body["usage"]["output_tokens"] == 5


async def test_agentic_os_mode_preserves_policy_prompt_and_tool_menu():
    fake = FakeOpenAI()
    fake.push([text_chunk("done"), finish_chunk("stop")])
    settings = Settings()
    settings.backend.base_url = "http://fake/v1"
    settings.pipeline.policy_owner = "agentic_os"
    settings.pipeline.max_tools = 1
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    body = request_body(
        stream=False,
        system="You are Claude Code\n\n## Agentic OS policy\nUse only the provided scope.",
        tools=[READ_TOOL, EDIT_TOOL, BASH_TOOL, GREP_TOOL],
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
        resp = await client.post("/v1/messages", json=body)
        stats = (await client.get("/stats")).json()

    assert resp.status_code == 200
    rendered = json.dumps(fake.requests[0])
    assert "You are Claude Code" in rendered
    assert "Agentic OS policy" in rendered
    assert "expert software engineering agent" not in rendered
    assert len(fake.requests[0]["tools"]) == 4
    assert stats["runtime"]["latest_pipeline_tool_count"] == 4


async def test_agentic_os_mode_disables_post_response_memory_capture():
    fake = FakeOpenAI()
    fake.push([text_chunk("done"), finish_chunk("stop")])
    settings = Settings()
    settings.backend.base_url = "http://fake/v1"
    settings.pipeline.policy_owner = "agentic_os"
    settings.memory.enabled = True
    settings.memory.idle_s = 0
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    body = request_body(stream=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
        resp = await client.post("/v1/messages", json=body)
        await asyncio.sleep(0.05)

    assert resp.status_code == 200
    assert len(fake.requests) == 1


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
    assert resp.json()["runtime"]["invalid_tool_rate_pct"] == 0.0
    assert resp.json()["runtime"]["latest_client_tool_count"] == 1
    assert resp.json()["runtime"]["latest_pipeline_tool_count"] == 1
    assert resp.json()["runtime"]["latest_backend_tool_count"] == 1


async def test_stats_records_preflight_success_outcome():
    fake = FakeOpenAI()
    fake.push([
        tool_chunk("c1", "Read", '{"file_path": "/work/old-root/src/main.c"}'),
        finish_chunk("tool_calls"),
    ])
    fake.push([text_chunk("done"), finish_chunk("stop")])
    body = request_body(stream=False, tools=[READ_TOOL])
    async with make_client(fake, path_aliases=[["/work/old-root", "/work/new-root"]]) as client:
        first = await client.post("/v1/messages", json=body)
        assert first.status_code == 200
        followup = request_body(stream=False, tools=[READ_TOOL])
        followup["messages"] = [
            body["messages"][0],
            {
                "role": "assistant",
                "content": [{
                    "id": "c1",
                    "type": "tool_use",
                    "name": "Read",
                    "input": {"file_path": "/work/new-root/src/main.c"},
                }],
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "c1",
                    "content": "ok",
                }],
            },
        ]
        await client.post("/v1/messages", json=followup)
        stats = (await client.get("/stats")).json()
    assert stats["runtime"]["preflight_rewrites"] == 1
    assert stats["runtime"]["tool_success_after_preflight"] == 1
    assert stats["runtime"]["tool_failure_after_preflight"] == 0


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


async def test_path_alias_canonicalized_before_backend_prompt():
    body = request_body(stream=False, tools=[READ_TOOL])
    body["messages"] = [{"role": "user", "content": "Read /work/old-root/src/main.c"}]
    fake = FakeOpenAI()
    fake.push([text_chunk("ok"), finish_chunk("stop")])
    async with make_client(fake, path_aliases=[["/work/old-root", "/work/new-root"]]) as client:
        resp = await client.post("/v1/messages", json=body)
    assert resp.status_code == 200
    rendered = json.dumps(fake.requests[0])
    assert "/work/old-root" not in rendered
    assert "/work/new-root/src/main.c" in rendered


async def test_planning_scaffold_generated_once_and_injected():
    fake = FakeOpenAI()
    fake.push([
        text_chunk("1. Inspect the failing test\n2. Patch the implementation\n3. Run the tests"),
        finish_chunk("stop"),
    ])
    fake.push([text_chunk("ok"), finish_chunk("stop")])
    fake.push([text_chunk("ok again"), finish_chunk("stop")])

    settings = Settings()
    settings.backend.base_url = "http://fake/v1"
    settings.planning.enabled = True
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")

    async with client:
        resp = await client.post("/v1/messages", json=request_body(stream=False, tools=[]))
        assert resp.status_code == 200
        followup = request_body(stream=False, tools=[])
        resp = await client.post("/v1/messages", json=followup)
        assert resp.status_code == 200

    assert len(fake.requests) == 3
    assert "Write a concrete execution plan" in fake.requests[0]["messages"][0]["content"]
    first_exec = fake.requests[1]["messages"][0]["content"]
    second_exec = fake.requests[2]["messages"][0]["content"]
    assert "## Execution plan" in first_exec
    assert "1. Inspect the failing test" in first_exec
    assert "Plan status: Step 1/3" in first_exec
    assert "## Execution plan" in second_exec
    assert "Write a concrete execution plan" not in json.dumps(fake.requests[2])


async def test_reasoning_route_uses_readonly_tools_only():
    from harness.config import PoolBackendCfg

    fake = FakeOpenAI()
    fake.push([text_chunk("explanation"), finish_chunk("stop")])

    settings = Settings()
    settings.backends = [
        PoolBackendCfg(
            name="reasoner",
            base_url="http://fake/v1",
            model="r",
            roles=["reasoning"],
        ),
        PoolBackendCfg(
            name="executor",
            base_url="http://fake/v1",
            model="m",
            roles=["main"],
        ),
    ]
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    body = request_body(stream=False, tools=[READ_TOOL, EDIT_TOOL])
    body["messages"] = [{"role": "user", "content": "Explain how /x works"}]

    async with client:
        resp = await client.post("/v1/messages", json=body)

    assert resp.status_code == 200
    sent = fake.requests[0]
    assert sent["model"] == "r"
    assert [t["function"]["name"] for t in sent.get("tools", [])] == ["Read"]


async def test_reasoning_route_can_still_surface_an_unsurfaced_tool():
    """The readonly reasoning filter narrows what the model SEES; it must not
    narrow what the model may CALL. Live 2026-07-29: an MCP-server turn worded
    "explain ..." routed to reasoning, the filter overwrote all_tools with the
    5-name readonly allowlist, and _surface_tool could no longer recover the
    MCP tool — every attempt died as "unknown tool ...; available tools: Read,
    WebFetch" until the retry budget ran out."""
    from harness.config import PoolBackendCfg

    mcp_tool = {
        "name": "mcp__kaibo__consult",
        "description": "Ask another model about the codebase",
        "input_schema": {
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
        },
    }

    fake = FakeOpenAI()
    fake.push([
        tool_chunk("c1", "mcp__kaibo__consult", '{"prompt": "how does relay work"}'),
        finish_chunk("tool_calls"),
    ])

    settings = Settings()
    settings.backends = [
        PoolBackendCfg(name="reasoner", base_url="http://fake/v1", model="r", roles=["reasoning"]),
        PoolBackendCfg(name="executor", base_url="http://fake/v1", model="m", roles=["main"]),
    ]
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    body = request_body(stream=False, tools=[READ_TOOL, EDIT_TOOL, mcp_tool])
    body["messages"] = [{"role": "user", "content": "Explain what kaibo says about the relay"}]

    async with client:
        resp = await client.post("/v1/messages", json=body)

    assert resp.status_code == 200
    payload = resp.json()
    text = json.dumps(payload)
    assert "unknown tool" not in text
    assert "[harness]" not in text
    calls = [b for b in payload["content"] if b["type"] == "tool_use"]
    assert [c["name"] for c in calls] == ["mcp__kaibo__consult"]


async def test_review_sidecar_adds_feedback_on_done_guard():
    from harness.config import PoolBackendCfg

    fake = FakeOpenAI()
    fake.push([text_chunk("done"), finish_chunk("stop")])
    fake.push([text_chunk("Run pytest -q before claiming completion."), finish_chunk("stop")])
    fake.push([tool_chunk("b1", "Bash", '{"command": "pytest -q"}'), finish_chunk("tool_calls")])

    settings = Settings()
    settings.review.enabled = True
    settings.backends = [
        PoolBackendCfg(
            name="executor",
            base_url="http://fake/v1",
            model="m",
            roles=["main"],
        ),
        PoolBackendCfg(
            name="critic",
            base_url="http://fake/v1",
            model="r",
            roles=["review"],
        ),
    ]
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    body = request_body(stream=False, tools=[READ_TOOL, EDIT_TOOL, {
        "name": "Bash",
        "description": "Runs a command",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }])
    body["messages"] = [
        {"role": "user", "content": "Fix /x"},
        {
            "role": "assistant",
            "content": [{
                "id": "e1",
                "type": "tool_use",
                "name": "Edit",
                "input": {
                        "file_path": "/x",
                        "old_string": "a",
                        "new_string": "b",
                },
            }],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "e1", "content": "edited"}]},
    ]

    async with client:
        resp = await client.post("/v1/messages", json=body)

    assert resp.status_code == 200
    assert len(fake.requests) == 3
    assert fake.requests[1]["model"] == "r"
    assert "runtime critic" in fake.requests[1]["messages"][0]["content"]
    assert "Reviewer feedback" in json.dumps(fake.requests[2])
    assert "Run pytest -q" in json.dumps(fake.requests[2])


async def test_backend_relaxed_planning_skips_planner():
    from harness.config import PoolBackendCfg

    fake = FakeOpenAI()
    fake.push([text_chunk("ok"), finish_chunk("stop")])

    settings = Settings()
    settings.planning.enabled = True
    settings.backends = [
        PoolBackendCfg(
            name="ready",
            base_url="http://fake/v1",
            model="m",
            roles=["main", "subagent", "fast"],
            relaxed=["planning"],
        )
    ]
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")

    async with client:
        resp = await client.post("/v1/messages", json=request_body(stream=False, tools=[]))

    assert resp.status_code == 200
    assert len(fake.requests) == 1
    sent = fake.requests[0]["messages"][0]["content"]
    assert "Write a concrete execution plan" not in sent
    assert "## Execution plan" not in sent


async def test_backend_relaxed_edit_guard_allows_direct_edit():
    from harness.config import PoolBackendCfg

    fake = FakeOpenAI()
    fake.push([
        tool_chunk("e1", "Edit", '{"file_path": "/x", "old_string": "a", "new_string": "b"}'),
        finish_chunk("tool_calls"),
    ])

    settings = Settings()
    settings.backends = [
        PoolBackendCfg(
            name="ready",
            base_url="http://fake/v1",
            model="m",
            roles=["main", "subagent", "fast"],
            relaxed=["guard_edit_without_read"],
        )
    ]
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")

    async with client:
        resp = await client.post(
            "/v1/messages",
            json=request_body(stream=True, tools=[EDIT_TOOL]),
        )

    assert resp.status_code == 200
    assert len(fake.requests) == 1
    evs = sse_events(resp.text)
    tool_start = next(
        d for n, d in evs
        if n == "content_block_start" and d["content_block"]["type"] == "tool_use"
    )
    assert tool_start["content_block"]["name"] == "Edit"


async def test_research_brief_generated_cached_and_injected(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("The service uses pytest -q and stores handlers in app.py.")
    fake = FakeOpenAI()
    fake.push([text_chunk("- Use pytest -q\n- Handler code lives in app.py"), finish_chunk("stop")])
    fake.push([text_chunk("ok"), finish_chunk("stop")])
    fake.push([text_chunk("ok cached"), finish_chunk("stop")])

    settings = Settings()
    settings.backend.base_url = "http://fake/v1"
    settings.research.enabled = True
    settings.research.cache_dir = str(tmp_path / "research-cache")
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    body = request_body(stream=False)
    body["messages"] = [{"role": "user", "content": f"research: file://{source}\nUse it."}]

    async with client:
        assert (await client.post("/v1/messages", json=body)).status_code == 200
        assert (await client.post("/v1/messages", json=body)).status_code == 200

    assert len(fake.requests) == 3
    assert "Summarize this research source" in fake.requests[0]["messages"][0]["content"]
    assert "## Research brief" in fake.requests[1]["messages"][0]["content"]
    assert "Handler code lives in app.py" in fake.requests[1]["messages"][0]["content"]
    assert "Summarize this research source" not in json.dumps(fake.requests[2])
    assert "## Research brief" in fake.requests[2]["messages"][0]["content"]


async def test_research_brief_records_project_memory_fact(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("The durable fact is pytest -q.")
    fake = FakeOpenAI()
    fake.push([text_chunk("- Use pytest -q for verification"), finish_chunk("stop")])
    fake.push([text_chunk("ok"), finish_chunk("stop")])
    settings = Settings()
    settings.backend.base_url = "http://fake/v1"
    settings.research.enabled = True
    settings.research.cache_dir = str(tmp_path / "research-cache")
    settings.memory.enabled = True
    settings.memory.dir = str(tmp_path / "memory")
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    body = request_body(stream=False)
    body["system"] = "Working directory: /repo"
    body["messages"] = [{"role": "user", "content": f"research: file://{source}\nUse it."}]
    async with client:
        assert (await client.post("/v1/messages", json=body)).status_code == 200
    mem = (tmp_path / "memory" / "repo.md").read_text()
    assert "research " in mem
    assert "Use pytest -q" in mem


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
        resp = await client.post("/v1/messages", json=request_body(stream=False, tools=[]))
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


async def test_stats_rehydrated_from_rotated_request_logs(tmp_path):
    rotated = tmp_path / "requests-20260619-100.jsonl"
    current = tmp_path / "requests.jsonl"
    rotated.write_text(json.dumps({
        "backend": "default", "input_tokens": 100, "output_tokens": 10,
        "cached_tokens": 20, "ttft_ms": 1,
    }) + "\n")
    current.write_text(json.dumps({
        "backend": "default", "input_tokens": 50, "output_tokens": 5,
        "cached_tokens": 0, "ttft_ms": 2,
    }) + "\n")

    settings = Settings()
    settings.backend.base_url = "http://fake/v1"
    settings.log.requests_path = str(current)
    fake = FakeOpenAI()
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    async with client:
        stats = (await client.get("/stats")).json()

    assert stats["input_tokens"] == 150
    assert stats["output_tokens"] == 15
    assert stats["cached_tokens"] == 20
    assert stats["backends"]["default"]["requests"] == 2


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


async def test_stats_tracks_vllm_decoded_tokens_and_live_rate():
    from harness.config import PoolBackendCfg

    fake = FakeOpenAI()
    fake.metrics_text = (
        'vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.42\n'
        'vllm:generation_tokens_total{engine="0",model_name="m"} 1000\n'
        'vllm:prompt_tokens_total{engine="0",model_name="m"} 5000\n'
        'vllm:prompt_tokens_cached_total{engine="0",model_name="m"} 3000\n'
    )
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
        first = (await client.get("/stats")).json()
        fake.metrics_text = (
            'vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.42\n'
            'vllm:generation_tokens_total{engine="0",model_name="m"} 1250\n'
            'vllm:prompt_tokens_total{engine="0",model_name="m"} 5100\n'
            'vllm:prompt_tokens_cached_total{engine="0",model_name="m"} 3050\n'
        )
        second = (await client.get("/stats")).json()

    assert first["vllm_decoded_tokens"] == 1000
    assert first["backends"]["v"]["vllm_decoded_tokens"] == 1000
    assert first["live_output_tps"] is None
    assert second["vllm_decoded_tokens"] == 1250
    assert second["vllm_prompt_tokens"] == 5100
    assert second["vllm_cached_prompt_tokens"] == 3050
    assert second["live_output_tps"] is not None
    assert second["live_output_tps"] > 0
    assert second["backends"]["v"]["live_output_tps"] == second["live_output_tps"]


async def test_vllm_decoded_tokens_persist_across_harness_and_vllm_restarts(tmp_path):
    from harness.config import PoolBackendCfg

    def metrics(output: int, prompt: int = 5000, cached: int = 3000) -> str:
        return "\n".join([
            'vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.42',
            f'vllm:generation_tokens_total{{engine="0",model_name="m"}} {output}',
            f'vllm:prompt_tokens_total{{engine="0",model_name="m"}} {prompt}',
            f'vllm:prompt_tokens_cached_total{{engine="0",model_name="m"}} {cached}',
        ])

    settings = Settings()
    settings.log.requests_path = str(tmp_path / "requests.jsonl")
    settings.backends = [
        PoolBackendCfg(name="v", kind="vllm", base_url="http://fake/v1",
                       model="m", roles=["main", "subagent", "fast"]),
    ]
    fake = FakeOpenAI()
    fake.metrics_text = metrics(1000)
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    async with client:
        first = (await client.get("/stats")).json()
        fake.metrics_text = metrics(1250)
        second = (await client.get("/stats")).json()

    assert first["vllm_decoded_tokens"] == 1000
    assert second["vllm_decoded_tokens"] == 1250

    fake.metrics_text = metrics(1500)
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    async with client:
        after_harness_restart = (await client.get("/stats")).json()
        fake.metrics_text = metrics(80)  # vLLM restarted; Prometheus counter reset
        after_vllm_restart = (await client.get("/stats")).json()

    assert after_harness_restart["vllm_decoded_tokens"] == 1500
    assert after_harness_restart["live_output_tps"] is None
    assert after_vllm_restart["vllm_decoded_tokens"] == 1580
    persisted = json.loads((tmp_path / "stats_state.json").read_text())
    assert persisted["vllm_totals"]["v"]["output"] == 1580
    assert persisted["vllm_last_counters"]["v"]["output"] == 80


async def test_vllm_totals_freeze_under_retired_model_on_swap(tmp_path):
    # A backend's `model` in config can change (new weights swapped onto the
    # same base_url) while its counters keep accumulating under the same
    # backend name. Cost is priced by model id, so tokens generated under the
    # old model must stay attributed to the old model id forever, not get
    # silently re-rated at whatever price the new model carries.
    from harness.config import PoolBackendCfg

    def metrics(output: int, prompt: int = 5000, cached: int = 3000) -> str:
        return "\n".join([
            'vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.42',
            f'vllm:generation_tokens_total{{engine="0",model_name="m"}} {output}',
            f'vllm:prompt_tokens_total{{engine="0",model_name="m"}} {prompt}',
            f'vllm:prompt_tokens_cached_total{{engine="0",model_name="m"}} {cached}',
        ])

    settings = Settings()
    settings.log.requests_path = str(tmp_path / "requests.jsonl")
    settings.backends = [
        PoolBackendCfg(name="v", kind="vllm", base_url="http://fake/v1",
                       model="old-model", roles=["main", "subagent", "fast"]),
    ]
    fake = FakeOpenAI()
    fake.metrics_text = metrics(1000)
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    async with client:
        await client.get("/stats")
        fake.metrics_text = metrics(1250)
        before_swap = (await client.get("/stats")).json()

    assert before_swap["backends"]["v"]["vllm_decoded_tokens"] == 1250

    # Model swapped: same backend name "v", new model, new vLLM process (its
    # own fresh Prometheus counter, unrelated to the old process's 1250).
    settings.backends = [
        PoolBackendCfg(name="v", kind="vllm", base_url="http://fake/v1",
                       model="new-model", roles=["main", "subagent", "fast"]),
    ]
    fake.metrics_text = metrics(80)
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    async with client:
        after_swap = (await client.get("/stats")).json()

    assert after_swap["backends"]["v"]["model"] == "new-model"
    # Fresh accumulation under the new model, NOT 1250 + 80.
    assert after_swap["backends"]["v"]["vllm_decoded_tokens"] == 80

    retired = after_swap["retired_backends"]
    assert len(retired) == 1
    frozen = next(iter(retired.values()))
    assert frozen["model"] == "old-model"
    assert frozen["vllm_decoded_tokens"] == 1250
    assert frozen["vllm_prompt_tokens"] == 5000
    assert frozen["vllm_cached_prompt_tokens"] == 3000

    persisted = json.loads((tmp_path / "stats_state.json").read_text())
    assert persisted["vllm_totals"]["v"]["output"] == 80
    retired_key = next(iter(persisted["retired_totals"]))
    assert persisted["retired_totals"][retired_key]["output"] == 1250
    assert persisted["retired_totals"][retired_key]["model"] == "old-model"


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


async def test_vllm_kv_used_uses_resident_estimate_as_floor(tmp_path):
    from harness.config import PoolBackendCfg

    log = tmp_path / "requests.jsonl"
    log.write_text(json.dumps({
        "backend": "v", "session_key": "sA", "input_tokens": 1000,
        "output_tokens": 500, "cached_tokens": 0, "ttft_ms": 1}) + "\n")
    fake = FakeOpenAI()
    fake.metrics_text = 'vllm:kv_cache_usage_perc{engine="0"} 0.0\n'
    settings = Settings()
    settings.log.requests_path = str(log)
    settings.backends = [
        PoolBackendCfg(name="v", kind="vllm", base_url="http://fake/v1",
                       model="m", roles=["main", "subagent", "fast"],
                       context_window=10000, max_in_flight=1),
    ]
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    async with client:
        d = (await client.get("/stats")).json()["backends"]["v"]
    assert d["kv_used_pct"] == 15.0
    assert d["kv_used_est"] is True


async def test_vllm_kv_used_prefers_higher_live_metric(tmp_path):
    from harness.config import PoolBackendCfg

    log = tmp_path / "requests.jsonl"
    log.write_text(json.dumps({
        "backend": "v", "session_key": "sA", "input_tokens": 1000,
        "output_tokens": 500, "cached_tokens": 0, "ttft_ms": 1}) + "\n")
    fake = FakeOpenAI()
    fake.metrics_text = 'vllm:kv_cache_usage_perc{engine="0"} 0.42\n'
    settings = Settings()
    settings.log.requests_path = str(log)
    settings.backends = [
        PoolBackendCfg(name="v", kind="vllm", base_url="http://fake/v1",
                       model="m", roles=["main", "subagent", "fast"],
                       context_window=10000, max_in_flight=1),
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


async def test_llamacpp_kv_used_prefers_live_slot_tokens(tmp_path):
    # while a slot is processing, /slots reports real token counts; the
    # estimate must never report less than what the engine shows live.
    from harness.config import PoolBackendCfg

    log = tmp_path / "requests.jsonl"
    log.write_text(json.dumps({
        "backend": "g", "session_key": "sA", "input_tokens": 100,
        "output_tokens": 0, "cached_tokens": 0, "ttft_ms": 1}) + "\n")

    fake = FakeOpenAI()
    fake.slots = [
        {"id": 0, "n_ctx": 1000, "is_processing": True, "n_prompt_tokens": 700},
        {"id": 1, "n_ctx": 1000, "is_processing": False},
    ]
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
    # live 700 > session estimate 100 -> 700 / 2000
    assert d["kv_used_pct"] == 35.0
    assert d["kv_used_est"] is True


async def test_kv_used_holds_last_reading_between_successful_polls():
    # a missed poll (busy backend, timeout) must not flicker the dashboard
    # to "-": /stats keeps serving the last good reading within its TTL.
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
        fake.metrics_text = None  # poll now fails (501)
        d = (await client.get("/stats")).json()["backends"]["v"]
    assert d["kv_used_pct"] == 42.0  # held, not flickered to null
    assert d["kv_used_est"] is False


# --- Claude Code compatibility: server-side tools and structured output ------

WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 8,
}


async def test_server_side_tool_definition_does_not_fail_the_request():
    # Claude Code's WebSearch is a separate Messages API call carrying this
    # definition; it has no input_schema, which used to 400 the whole request.
    fake = FakeOpenAI()
    fake.push([text_chunk("I cannot search the web here."), finish_chunk("stop")])
    async with make_client(fake) as client:
        resp = await client.post(
            "/v1/messages",
            json=request_body(stream=False, tools=[WEB_SEARCH_TOOL, READ_TOOL]),
        )
    assert resp.status_code == 200
    # the backend cannot execute a server-side tool, so it is never offered
    sent_tools = fake.requests[-1].get("tools") or []
    assert [t["function"]["name"] for t in sent_tools] == ["Read"]


async def test_server_side_tool_only_request_sends_no_tools():
    fake = FakeOpenAI()
    fake.push([text_chunk("no web access"), finish_chunk("stop")])
    async with make_client(fake) as client:
        resp = await client.post(
            "/v1/messages", json=request_body(stream=False, tools=[WEB_SEARCH_TOOL])
        )
    assert resp.status_code == 200
    assert "tools" not in fake.requests[-1]


async def test_count_tokens_accepts_server_side_tool():
    fake = FakeOpenAI()
    fake.push([text_chunk("x"), finish_chunk("stop")])
    async with make_client(fake) as client:
        resp = await client.post(
            "/v1/messages/count_tokens",
            json=request_body(stream=False, tools=[WEB_SEARCH_TOOL]),
        )
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] > 0


TITLE_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}},
    "required": ["title"],
    "additionalProperties": False,
}


async def test_output_config_schema_constrains_the_backend_request():
    fake = FakeOpenAI()
    fake.push([text_chunk('{"title": "fix the failing test"}'), finish_chunk("stop")])
    body = request_body(stream=False, tools=[])
    body["output_config"] = {
        "effort": "high",
        "format": {"type": "json_schema", "schema": TITLE_SCHEMA},
    }
    async with make_client(fake) as client:
        resp = await client.post("/v1/messages", json=body)
    assert resp.status_code == 200
    assert fake.requests[-1]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": TITLE_SCHEMA, "strict": True},
    }


async def test_request_without_output_config_sets_no_response_format():
    fake = FakeOpenAI()
    fake.push([text_chunk("plain"), finish_chunk("stop")])
    async with make_client(fake) as client:
        resp = await client.post("/v1/messages", json=request_body(stream=False, tools=[]))
    assert resp.status_code == 200
    assert "response_format" not in fake.requests[-1]


async def test_reasoning_route_shapes_tools_subtractively():
    """Blocked-list, never allow-list (2026-07-11 default-open-enforcement).
    The old allowlist named Grep/Glob/LS — tools that appear in 0 of 371 real
    requests — so it collapsed to Read+WebFetch and hid every MCP tool, which
    no allowlist in this codebase can enumerate."""
    from harness.config import PoolBackendCfg

    mcp_tool = {
        "name": "mcp__kaibo__consult",
        "description": "Ask another model about the codebase",
        "input_schema": {"type": "object", "properties": {"prompt": {"type": "string"}}},
    }
    fake = FakeOpenAI()
    fake.push([text_chunk("explanation"), finish_chunk("stop")])

    settings = Settings()
    settings.backends = [
        PoolBackendCfg(name="reasoner", base_url="http://fake/v1", model="r", roles=["reasoning"]),
    ]
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    body = request_body(stream=False, tools=[READ_TOOL, EDIT_TOOL, BASH_TOOL, mcp_tool])
    body["messages"] = [{"role": "user", "content": "Explain the relay"}]

    async with client:
        resp = await client.post("/v1/messages", json=body)

    assert resp.status_code == 200
    shown = [t["function"]["name"] for t in fake.requests[0].get("tools", [])]
    assert "mcp__kaibo__consult" in shown   # client-owned surface always passes
    assert "Read" in shown
    assert "Edit" not in shown              # the one thing the route exists to prevent


async def _payload_for(backend_cfg):
    """Route one request to a single-backend fleet and return what it sent."""
    fake = FakeOpenAI()
    fake.push([text_chunk("ok"), finish_chunk("stop")])

    settings = Settings()
    settings.backends = [backend_cfg]
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")

    async with client:
        resp = await client.post("/v1/messages", json=request_body(stream=False))

    assert resp.status_code == 200
    return fake.requests[0]


async def test_reasoning_capability_asks_a_thinking_backend_to_think():
    """The `reasoning` capability is the opt-in; the PROFILE owns the wire
    spelling (Law 4). This deployment's DeepSeek only emits a reasoning channel
    when sent chat_template_kwargs={"thinking": true} — probed 2026-07-30, it
    accepts thinking_token_budget and silently ignores it, so reasoning_budget
    (which owns the *depth* knob) cannot switch the channel on."""
    from harness.config import PoolBackendCfg

    sent = await _payload_for(
        PoolBackendCfg(
            name="thinker",
            base_url="http://fake/v1",
            model="r",
            profile="deepseek_r1",
            roles=["main", "subagent", "fast"],
            capabilities=["reasoning"],
        )
    )
    assert sent["chat_template_kwargs"] == {"thinking": True}


async def test_a_backend_without_the_reasoning_capability_is_left_alone():
    """Same thinking-capable profile, no declared capability: an undeclared
    backend's payload is untouched, so adding the profile never changes what an
    already-certified backend sends."""
    from harness.config import PoolBackendCfg

    sent = await _payload_for(
        PoolBackendCfg(
            name="quiet",
            base_url="http://fake/v1",
            model="r",
            profile="deepseek_r1",
            roles=["main", "subagent", "fast"],
        )
    )
    assert "chat_template_kwargs" not in sent


async def test_a_profile_with_no_thinking_switch_sends_nothing_extra():
    """Declaring the capability on a family whose profile has no thinking
    switch is inert rather than a guess: only a profile knows its own spelling
    (qwen3 says enable_thinking, this deepseek says thinking), so a family that
    has never been probed sends no invented field."""
    from harness.config import PoolBackendCfg

    sent = await _payload_for(
        PoolBackendCfg(
            name="qwenish",
            base_url="http://fake/v1",
            model="q",
            profile="qwen",
            roles=["main", "subagent", "fast"],
            capabilities=["reasoning"],
        )
    )
    assert "chat_template_kwargs" not in sent
