import json

import httpx

from harness.config import PoolBackendCfg, RiskProfileCfg, Settings
from harness.server import create_app
from tests.fake_openai import FakeOpenAI, finish_chunk, text_chunk, tool_chunk
from tests.test_server import BASH_TOOL, EDIT_TOOL, READ_TOOL, request_body

WRITE_TOOL = {
    "name": "Write",
    "description": "Writes a file",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["file_path", "content"],
    },
}


def critic_body():
    body = request_body(stream=False, tools=[EDIT_TOOL])
    body["max_tokens"] = 8192
    body["messages"] = [
        {"role": "user", "content": "Refactor drivers/net/foo.c"},
        {
            "role": "assistant",
            "content": [{
                "id": "e1",
                "type": "tool_use",
                "name": "Edit",
                "input": {
                    "file_path": "drivers/net/foo.c",
                    "old_string": "int old_sig(void)",
                    "new_string": "int new_sig(int flags)",
                },
            }],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "e1", "content": "edited"}]},
    ]
    return body


def settings(tmp_path=None):
    s = Settings()
    s.critic.enabled = True
    s.reasoning_budget.enabled = True
    s.pipeline.path_aliases = [["/work/old-root", "/work/new-root"]]
    if tmp_path:
        s.log.requests_path = str(tmp_path / "requests.jsonl")
    s.risk_profiles = [
        RiskProfileCfg(
            name="kernel",
            path_patterns=["drivers/**"],
            text_patterns=["spinlock"],
            plan_mode="kernel_change_plan",
            critic_mode="kernel_critic",
        )
    ]
    s.backends = [
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
            roles=["critic"],
            capabilities=["reasoning_budget"],
        ),
    ]
    return s


async def test_critic_injects_feedback_before_executor(tmp_path):
    feedback = (
        "REVISE: update all callers and add a build check.\n"
        + "Keep this full feedback for fine tuning. " * 80
    )
    fake = FakeOpenAI()
    fake.push([
        {"choices": [{"index": 0, "delta": {"reasoning": "check ABI"}, "finish_reason": None}]},
        text_chunk(feedback),
        finish_chunk("stop"),
    ])
    fake.push([text_chunk("I'll update callers."), finish_chunk("stop")])
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings(tmp_path), backend_client=backend_client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
        resp = await client.post("/v1/messages", json=critic_body())
        stats = (await client.get("/stats")).json()
    assert resp.status_code == 200
    assert fake.requests[0]["model"] == "r"
    assert fake.requests[0]["thinking_token_budget"] == 4096
    assert fake.requests[1]["model"] == "m"
    assert "Critic feedback" in json.dumps(fake.requests[1])
    assert "update all callers" in json.dumps(fake.requests[1])
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    sidecar = next(r for r in rows if r.get("sidecar_type") == "critic")
    assert sidecar["critic_action"] == "revise"
    assert sidecar["critic_matched_profiles"] == ["kernel"]
    assert sidecar["critic_feedback"] == feedback
    assert sidecar["critic_feedback_hash"]
    assert "verification_gap" in sidecar["critic_feedback_tags"]
    assert sidecar["input_tokens"] == 10
    assert sidecar["output_tokens"] == 5
    assert sidecar["cached_tokens"] == 0
    assert sidecar["stop_reason"] == "end_turn"
    assert sidecar["ttft_ms"] >= 0
    assert sidecar["reasoning_tokens_observed"] > 0
    assert stats["backends"]["critic"]["kv_written_tokens"] == 15
    assert stats["backends"]["critic"]["requests"] == 1
    assert stats["backends"]["critic"]["ttft_p50_ms"] >= 0
    assert stats["critic"]["calls"] == 1
    assert stats["critic"]["revise"] == 1
    assert stats["critic"]["recent_revise"] == 1
    assert stats["critic"]["feedback_tags"]["verification_gap"] == 1
    assert stats["critic"]["feedback_hashes"][sidecar["critic_feedback_hash"]] == 1


async def test_critic_approve_does_not_inject_feedback(tmp_path):
    fake = FakeOpenAI()
    fake.push([text_chunk("APPROVE"), finish_chunk("stop")])
    fake.push([text_chunk("continuing"), finish_chunk("stop")])
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings(tmp_path), backend_client=backend_client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
        resp = await client.post("/v1/messages", json=critic_body())
    assert resp.status_code == 200
    assert "Critic feedback" not in json.dumps(fake.requests[1])
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    sidecar = next(r for r in rows if r.get("sidecar_type") == "critic")
    assert sidecar["critic_action"] == "approve"


async def test_agentic_os_mode_runs_enabled_local_critic(tmp_path):
    fake = FakeOpenAI()
    fake.push([text_chunk("APPROVE"), finish_chunk("stop")])
    fake.push([text_chunk("continuing"), finish_chunk("stop")])
    s = settings(tmp_path)
    s.pipeline.policy_owner = "agentic_os"
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(s, backend_client=backend_client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
        resp = await client.post("/v1/messages", json=critic_body())
        stats = (await client.get("/stats")).json()

    assert resp.status_code == 200
    assert len(fake.requests) == 2
    assert fake.requests[0]["model"] == "r"
    assert fake.requests[1]["model"] == "m"
    assert stats["critic"]["calls"] == 1
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    sidecar = next(r for r in rows if r.get("sidecar_type") == "critic")
    assert sidecar["critic_policy_owner"] == "agentic_os"
    assert sidecar["critic_forced"] is False


async def test_critic_enabled_steers_runtime_preflight_failures_in_agentic_os(tmp_path):
    fake = FakeOpenAI()
    fake.push([
        tool_chunk(
            "b1",
            "Bash",
            '{"command": "cat > /tmp/check_task.sh << \\"EOF\\"\\necho hi\\nEOF\\nbash /tmp/check_task.sh"}',
        ),
        finish_chunk("tool_calls"),
    ])
    fake.push([
        text_chunk("Use Write with file_path and content; do not create files with Bash heredocs."),
        finish_chunk("stop"),
    ])
    fake.push([
        tool_chunk(
            "w1",
            "Write",
            '{"file_path": "/tmp/check_task.sh", "content": "echo hi"}',
        ),
        finish_chunk("tool_calls"),
    ])
    s = settings(tmp_path)
    s.pipeline.policy_owner = "agentic_os"
    s.review.enabled = False
    s.critic.max_tokens = 32768
    body = request_body(stream=False, tools=[WRITE_TOOL, BASH_TOOL])
    body["messages"] = [{"role": "user", "content": "create check_task.sh"}]
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(s, backend_client=backend_client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
        resp = await client.post("/v1/messages", json=body)
        stats = (await client.get("/stats")).json()

    assert resp.status_code == 200
    assert [r["model"] for r in fake.requests] == ["m", "r", "m"]
    assert fake.requests[1]["max_tokens"] == 32768
    assert "Reviewer feedback" in json.dumps(fake.requests[2])
    assert "Bash heredocs" in json.dumps(fake.requests[2])
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    sidecar = next(r for r in rows if r.get("sidecar_type") == "review")
    assert sidecar["review_trigger"] == "use_write_tool"
    assert sidecar["review_action"] == "revise"
    assert sidecar["model"] == "r"
    runtime_record = next(r for r in rows if r.get("kind") != "sidecar")
    assert runtime_record["preflight_reasons"]["use_write_tool"] == 1
    assert runtime_record["review_trigger"] == "use_write_tool"
    assert stats["critic"]["calls"] == 0
    assert stats["runtime"]["preflight_reasons"]["use_write_tool"] == 1


async def test_runtime_review_max_tokens_fails_closed(tmp_path):
    fake = FakeOpenAI()
    fake.push([
        tool_chunk(
            "b1",
            "Bash",
            '{"command": "cat > /tmp/check_task.sh << \\"EOF\\"\\necho hi\\nEOF"}',
        ),
        finish_chunk("tool_calls"),
    ])
    fake.push([finish_chunk("length")])
    fake.push([
        tool_chunk(
            "w1",
            "Write",
            '{"file_path": "/tmp/check_task.sh", "content": "echo hi"}',
        ),
        finish_chunk("tool_calls"),
    ])
    s = settings(tmp_path)
    s.pipeline.policy_owner = "agentic_os"
    s.review.enabled = False
    body = request_body(stream=False, tools=[WRITE_TOOL, BASH_TOOL])
    body["messages"] = [{"role": "user", "content": "create check_task.sh"}]
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(s, backend_client=backend_client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
        resp = await client.post("/v1/messages", json=body)

    assert resp.status_code == 200
    assert [r["model"] for r in fake.requests] == ["m", "r", "m"]
    assert "Reviewer feedback" in json.dumps(fake.requests[2])
    assert "hit its token limit" in json.dumps(fake.requests[2])
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    sidecar = next(r for r in rows if r.get("sidecar_type") == "review")
    assert sidecar["review_trigger"] == "use_write_tool"
    assert sidecar["review_action"] == "inconclusive"
    assert sidecar["review_inconclusive_reason"] == "max_tokens_without_feedback"
    runtime_record = next(r for r in rows if r.get("kind") != "sidecar")
    assert runtime_record["review_action"] == "inconclusive"
    assert runtime_record["review_generated"] == 1


async def test_agentic_os_mode_runs_critic_when_requested(tmp_path):
    fake = FakeOpenAI()
    fake.push([text_chunk("REVISE: verify the ABI update."), finish_chunk("stop")])
    fake.push([text_chunk("I will verify it."), finish_chunk("stop")])
    s = settings(tmp_path)
    s.pipeline.policy_owner = "agentic_os"
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(s, backend_client=backend_client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
        resp = await client.post(
            "/v1/messages",
            json=critic_body(),
            headers={"x-agentic-critic": "required"},
        )

    assert resp.status_code == 200
    assert fake.requests[0]["model"] == "r"
    assert fake.requests[1]["model"] == "m"
    assert "Critic feedback" in json.dumps(fake.requests[1])
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    sidecar = next(r for r in rows if r.get("sidecar_type") == "critic")
    assert sidecar["critic_policy_owner"] == "agentic_os"
    assert sidecar["critic_forced"] is True


async def test_critic_max_tokens_empty_approval_is_inconclusive(tmp_path):
    fake = FakeOpenAI()
    fake.push([text_chunk("APPROVE"), finish_chunk("length")])
    fake.push([text_chunk("continuing"), finish_chunk("stop")])
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings(tmp_path), backend_client=backend_client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
        resp = await client.post("/v1/messages", json=critic_body())
        stats = (await client.get("/stats")).json()
    assert resp.status_code == 200
    assert "Critic feedback" not in json.dumps(fake.requests[1])
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    sidecar = next(r for r in rows if r.get("sidecar_type") == "critic")
    assert sidecar["critic_action"] == "inconclusive"
    assert sidecar["critic_inconclusive_reason"] == "max_tokens_without_feedback"
    assert stats["critic"]["inconclusive"] == 1
    assert stats["critic"]["inconclusive_reasons"]["max_tokens_without_feedback"] == 1


async def test_critic_degrades_without_backend():
    fake = FakeOpenAI()
    fake.push([text_chunk("continuing"), finish_chunk("stop")])
    s = settings()
    s.backends = [
        PoolBackendCfg(
            name="executor",
            base_url="http://fake/v1",
            model="m",
            roles=["main"],
        )
    ]
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(s, backend_client=backend_client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
        resp = await client.post("/v1/messages", json=critic_body())
    assert resp.status_code == 200
    assert len(fake.requests) == 1


async def test_critic_skips_deterministic_path_alias(tmp_path):
    body = request_body(stream=False, tools=[READ_TOOL])
    body["messages"] = [
        {"role": "user", "content": "read the file"},
        {
            "role": "assistant",
            "content": [{
                "id": "r1",
                "type": "tool_use",
                "name": "Read",
                "input": {"file_path": "/work/old-root/src/main.c"},
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "r1",
                "is_error": True,
                "content": "No such file or directory: /work/old-root/src/main.c",
            }],
        },
    ]
    fake = FakeOpenAI()
    fake.push([text_chunk("continuing"), finish_chunk("stop")])
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings(tmp_path), backend_client=backend_client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
        resp = await client.post("/v1/messages", json=body)
        stats = (await client.get("/stats")).json()
    assert resp.status_code == 200
    assert len(fake.requests) == 1
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    record = rows[0]
    assert record["critic_eligible"] is False
    assert record["critic_skipped_reason"] == "path_alias"
    assert record["critic_saved_turn_estimate"] == 1
    assert stats["critic"]["calls"] == 0
    assert stats["runtime"]["critic_skips"]["path_alias"] == 1


async def test_critic_skips_gateguard_fact_force(tmp_path):
    body = request_body(stream=False, tools=[EDIT_TOOL])
    body["messages"] = [
        {"role": "user", "content": "Refactor drivers/net/foo.c"},
        {
            "role": "assistant",
            "content": [{
                "id": "e1",
                "type": "tool_use",
                "name": "Edit",
                "input": {
                    "file_path": "drivers/net/foo.c",
                    "old_string": "int old_sig(void)",
                    "new_string": "int new_sig(int flags)",
                },
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "e1",
                "is_error": True,
                "content": "ERROR: [Fact-Forcing Gate]\n\nBefore editing drivers/net/foo.c, present these facts.",
            }],
        },
    ]
    fake = FakeOpenAI()
    fake.push([text_chunk("continuing"), finish_chunk("stop")])
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings(tmp_path), backend_client=backend_client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
        resp = await client.post("/v1/messages", json=body)
        stats = (await client.get("/stats")).json()
    assert resp.status_code == 200
    assert len(fake.requests) == 1
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    record = rows[0]
    assert record["critic_eligible"] is False
    assert record["critic_skipped_reason"] == "gateguard_fact_force"
    assert stats["critic"]["calls"] == 0
    assert stats["runtime"]["critic_skips"]["gateguard_fact_force"] == 1
