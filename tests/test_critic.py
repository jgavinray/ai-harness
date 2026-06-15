import json

import httpx

from harness.config import PoolBackendCfg, RiskProfileCfg, Settings
from harness.server import create_app
from tests.fake_openai import FakeOpenAI, finish_chunk, text_chunk
from tests.test_server import EDIT_TOOL, request_body


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
    fake = FakeOpenAI()
    fake.push([
        {"choices": [{"index": 0, "delta": {"reasoning": "check ABI"}, "finish_reason": None}]},
        text_chunk("REVISE: update all callers and add a build check."),
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
    assert sidecar["input_tokens"] == 10
    assert sidecar["output_tokens"] == 5
    assert sidecar["cached_tokens"] == 0
    assert sidecar["stop_reason"] == "end_turn"
    assert sidecar["reasoning_tokens_observed"] > 0
    assert stats["backends"]["critic"]["kv_written_tokens"] == 15
    assert stats["backends"]["critic"]["requests"] == 1


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
