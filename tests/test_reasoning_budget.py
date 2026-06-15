import json

import httpx

from harness.config import PoolBackendCfg, RiskProfileCfg, Settings
from harness.ir import Conversation, GenParams
from harness.reasoning_budget import decide
from harness.server import create_app
from tests.fake_openai import FakeOpenAI, finish_chunk, text_chunk
from tests.test_server import request_body


class BackendStub:
    def __init__(self, capabilities=None, in_flight=0):
        self.cfg = type("Cfg", (), {"capabilities": capabilities or []})()
        self.in_flight = in_flight


def conv(max_tokens=8192):
    return Conversation("", (), (), GenParams(max_tokens=max_tokens))


def settings():
    s = Settings()
    s.reasoning_budget.enabled = True
    return s


def test_reasoning_budget_skips_backend_without_capability():
    d = decide(settings(), BackendStub([]), "reasoning", {}, conv())
    assert d.budget is None
    assert d.skipped_reason == "backend_lacks_capability"


def test_reasoning_budget_applies_and_clamps_to_final_answer_reserve():
    d = decide(settings(), BackendStub(["reasoning_budget"]), "reasoning", {}, conv(max_tokens=3000))
    assert d.budget == 1500
    assert d.clamped_by == "final_answer_reserve"


def test_reasoning_budget_risk_profile_escalates_then_auto_clamps():
    s = settings()
    s.risk_profiles = [
        RiskProfileCfg(
            name="kernel",
            path_patterns=["drivers/**", "include/linux/**"],
            text_patterns=["spinlock"],
            plan_mode="kernel_change_plan",
            critic_mode="kernel_critic",
        )
    ]
    body = {"messages": [{"role": "user", "content": "Plan refactor for drivers/net/foo.c"}]}
    d = decide(s, BackendStub(["reasoning_budget"]), "plan", body, conv(max_tokens=20000))
    assert d.mode == "kernel_change_plan"
    assert d.budget == s.reasoning_budget.max_auto_tokens
    assert d.matched_profiles == ["kernel"]


def test_reasoning_budget_load_sheds():
    d = decide(settings(), BackendStub(["reasoning_budget"], in_flight=1), "plan", {}, conv(max_tokens=20000))
    assert d.budget == 2048
    assert d.clamped_by == "load_shed"


async def test_server_sends_budget_only_to_capable_backend_and_logs(tmp_path):
    fake = FakeOpenAI()
    fake.push([
        {"choices": [{"index": 0, "delta": {"reasoning": "thinking about it"}, "finish_reason": None}]},
        text_chunk("done"),
        finish_chunk("stop"),
    ])
    s = settings()
    s.log.requests_path = str(tmp_path / "requests.jsonl")
    s.backends = [
        PoolBackendCfg(
            name="reasoner",
            base_url="http://fake/v1",
            model="r",
            roles=["reasoning"],
            capabilities=["reasoning_budget"],
        )
    ]
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(s, backend_client=backend_client)
    body = request_body(stream=False)
    body["max_tokens"] = 8192
    body["messages"] = [{"role": "user", "content": "Explain the architecture tradeoff"}]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
        resp = await client.post("/v1/messages", json=body)
    assert resp.status_code == 200
    assert fake.requests[-1]["thinking_token_budget"] == 4096
    rec = json.loads((tmp_path / "requests.jsonl").read_text().splitlines()[-1])
    assert rec["reasoning_budget_sent"] == 4096
    assert rec["reasoning_budget_mode"] == "architecture_plan"
    assert rec["reasoning_tokens_observed"] > 0


async def test_server_does_not_send_budget_to_uncapable_backend():
    fake = FakeOpenAI()
    fake.push([text_chunk("done"), finish_chunk("stop")])
    s = settings()
    s.backends = [
        PoolBackendCfg(
            name="main",
            base_url="http://fake/v1",
            model="m",
            roles=["main", "reasoning"],
            capabilities=[],
        )
    ]
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(s, backend_client=backend_client)
    body = request_body(stream=False)
    body["messages"] = [{"role": "user", "content": "Explain the architecture tradeoff"}]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
        resp = await client.post("/v1/messages", json=body)
    assert resp.status_code == 200
    assert "thinking_token_budget" not in fake.requests[-1]
