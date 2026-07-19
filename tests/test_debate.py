"""Adversarial review debate (spec 2026-07-19-adversarial-review-loop-design.md).

Covers: consensus paths (approve / concede-regenerate / rebut-accepted),
pathology valves (no-progress, unchanged regen, deadline), fail-open on
reviewer error or malformed verdict, prefix-stable reviewer prompts,
keepalive pings, round logging, and shadow/enforce server wiring.
"""

import asyncio
import json

import httpx

from harness.config import PoolBackendCfg, Settings
from harness.debate import DebateManager, keepalive_iter
from harness.codec.anthropic_out import collect, stream_sse
from harness.ir import (
    Conversation,
    Done,
    GenParams,
    Ping,
    TextDelta,
    TextPart,
    ToolDef,
    Turn,
)
from harness.log import RequestLogger
from harness.profiles.registry import get_profile
from harness.server import create_app
from tests.fake_openai import FakeOpenAI, finish_chunk, text_chunk
from tests.test_server import request_body

OBJECTION = (
    "OBJECTION:\n1. The claim 'the answer is 42' cites no tool evidence "
    "in this conversation."
)


class ScriptedBackend:
    """Minimal PooledBackend stand-in: records payloads, streams scripted
    OpenAI chunk dicts (a script of [Exception] raises instead)."""

    def __init__(self, name: str, scripts: list) -> None:
        self.name = name
        self.model_name = "m"
        self.in_flight = 0
        self.profile = get_profile("qwen")
        self.scripts = scripts
        self.requests: list[dict] = []

    async def stream(self, payload: dict):
        self.requests.append(payload)
        script = self.scripts.pop(0) if len(self.scripts) > 1 else self.scripts[0]
        for chunk in script:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


class FakePool:
    def __init__(self, roles: dict[str, list]) -> None:
        self.roles = roles

    def with_role(self, role: str, include_down: bool = False) -> list:
        return self.roles.get(role, [])


class UsageRecorder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, backend, done, session_key=None, *, count_request=False, ttft_ms=None):
        self.calls.append({
            "backend": backend.name,
            "input_tokens": done.input_tokens,
            "count_request": count_request,
        })


READ_TOOL_DEF = ToolDef(
    name="Read",
    description="Reads a file",
    input_schema={"type": "object", "properties": {"file_path": {"type": "string"}}},
    original_schema={"type": "object", "properties": {"file_path": {"type": "string"}}},
)


def make_conv() -> Conversation:
    return Conversation(
        system="be brief",
        turns=(Turn("user", (TextPart("What is in config.py?"),)),),
        tools=(READ_TOOL_DEF,),
        params=GenParams(max_tokens=512),
    )


def candidate() -> list:
    return [TextDelta("The answer is 42."), Done("end_turn", 10, 5, 0)]


def script(text: str) -> list[dict]:
    return [text_chunk(text), finish_chunk("stop")]


def make_manager(tmp_path, **review_overrides) -> DebateManager:
    s = Settings()
    s.review.mode = "shadow"
    for key, value in review_overrides.items():
        setattr(s.review, key, value)
    logger = RequestLogger(None, directory=tmp_path / "reviews")
    return DebateManager(s, logger)


def revised_regen(calls: list):
    async def regen(regen_conv: Conversation) -> list:
        calls.append(regen_conv)
        return [
            TextDelta("Revised: the transcript has no evidence for config.py."),
            Done("end_turn", 12, 6, 0),
        ]
    return regen


def reviews_records(tmp_path) -> list[dict]:
    files = list((tmp_path / "reviews").glob("*.jsonl"))
    rows: list[dict] = []
    for f in files:
        rows += [json.loads(line) for line in f.read_text().splitlines()]
    return rows


# ---- unit: verdict outcomes ----


async def test_consensus_on_first_round_approve(tmp_path):
    reviewer = ScriptedBackend("rev", [script("APPROVE")])
    pool = FakePool({"review": [reviewer]})
    manager = make_manager(tmp_path)
    metrics: dict = {}
    outcome = await manager.run(
        make_conv(), candidate(), pool, metrics,
        session_key="s", parent_request_id="req1",
    )
    assert outcome.outcome == "consensus"
    assert outcome.rounds == 1
    assert outcome.events == candidate()
    assert outcome.unresolved_objection is None
    assert metrics["debate_outcome"] == "consensus"
    assert metrics["debate_rounds"] == 1


async def test_objection_concede_regenerates_to_consensus(tmp_path):
    reviewer = ScriptedBackend("rev", [
        script(OBJECTION),   # round 1: reviewer objects
        script("CONCEDE"),   # round 1: proposal concedes
        script("APPROVE"),   # round 2: revised candidate approved
    ])
    pool = FakePool({"review": [reviewer]})
    manager = make_manager(tmp_path)
    regen_calls: list = []
    usage = UsageRecorder()
    exec_backend = ScriptedBackend("exec", [[]])
    metrics: dict = {}
    outcome = await manager.run(
        make_conv(), candidate(), pool, metrics,
        backend=exec_backend, regenerate=revised_regen(regen_calls),
        session_key="s", parent_request_id="req1", account_usage=usage,
    )
    assert outcome.outcome == "consensus"
    assert outcome.rounds == 2
    assert any(isinstance(e, TextDelta) and "Revised" in e.text for e in outcome.events)
    assert len(regen_calls) == 1
    # the regen conversation carries the objection but is not the client conv
    regen_rendered = json.dumps(get_profile("qwen").render(regen_calls[0], "m"))
    assert "cites no tool evidence" in regen_rendered
    # discarded original candidate's usage is still accounted to the executor
    assert any(c["backend"] == "exec" and c["input_tokens"] == 10 for c in usage.calls)
    # reviewer + counter + reviewer sidecar calls are accounted as requests
    assert sum(1 for c in usage.calls if c["count_request"]) == 3


async def test_rebuttal_accepted_ships_original(tmp_path):
    reviewer = ScriptedBackend("rev", [
        script(OBJECTION),
        script("REBUT: the user message itself asks about config.py and 42 is quoted there."),
        script("APPROVE"),
    ])
    pool = FakePool({"review": [reviewer]})
    manager = make_manager(tmp_path)
    regen_calls: list = []
    outcome = await manager.run(
        make_conv(), candidate(), pool, {},
        regenerate=revised_regen(regen_calls),
        session_key="s", parent_request_id="req1",
    )
    assert outcome.outcome == "consensus"
    assert outcome.events == candidate()
    assert regen_calls == []


# ---- unit: pathology valves ----


async def test_repeated_objection_is_deadlock(tmp_path):
    reviewer = ScriptedBackend("rev", [
        script(OBJECTION),
        script("REBUT: the value is well known."),
        script(OBJECTION),  # identical objection: no progress
    ])
    pool = FakePool({"review": [reviewer]})
    manager = make_manager(tmp_path)
    metrics: dict = {}
    outcome = await manager.run(
        make_conv(), candidate(), pool, metrics,
        regenerate=revised_regen([]),
        session_key="s", parent_request_id="req1",
    )
    assert outcome.outcome == "deadlock"
    assert outcome.events == candidate()
    assert "cites no tool evidence" in outcome.unresolved_objection
    assert len(reviewer.requests) == 3


async def test_unchanged_regenerated_candidate_is_deadlock(tmp_path):
    reviewer = ScriptedBackend("rev", [script(OBJECTION), script("CONCEDE")])
    pool = FakePool({"review": [reviewer]})
    manager = make_manager(tmp_path)

    async def same_regen(regen_conv):
        return candidate()

    outcome = await manager.run(
        make_conv(), candidate(), pool, {},
        regenerate=same_regen,
        session_key="s", parent_request_id="req1",
    )
    assert outcome.outcome == "deadlock"
    assert outcome.events == candidate()


async def test_deadline_valve_ships_current_candidate(tmp_path):
    reviewer = ScriptedBackend("rev", [script("APPROVE")])
    pool = FakePool({"review": [reviewer]})
    manager = make_manager(tmp_path, client_deadline_s=0.0)
    outcome = await manager.run(
        make_conv(), candidate(), pool, {},
        session_key="s", parent_request_id="req1",
    )
    assert outcome.outcome == "deadline"
    assert outcome.events == candidate()
    assert outcome.rounds == 0
    assert reviewer.requests == []


# ---- unit: fail-open ----


async def test_reviewer_error_fails_open(tmp_path):
    reviewer = ScriptedBackend("rev", [[RuntimeError("backend died")]])
    pool = FakePool({"review": [reviewer]})
    manager = make_manager(tmp_path)
    metrics: dict = {}
    outcome = await manager.run(
        make_conv(), candidate(), pool, metrics,
        session_key="s", parent_request_id="req1",
    )
    assert outcome.outcome == "reviewer_error"
    assert outcome.events == candidate()


async def test_malformed_verdict_fails_open(tmp_path):
    reviewer = ScriptedBackend("rev", [script("Well, it depends on the file.")])
    pool = FakePool({"review": [reviewer]})
    manager = make_manager(tmp_path)
    outcome = await manager.run(
        make_conv(), candidate(), pool, {},
        session_key="s", parent_request_id="req1",
    )
    assert outcome.outcome == "reviewer_error"
    assert outcome.events == candidate()


async def test_no_review_backend_skips(tmp_path):
    manager = make_manager(tmp_path)
    outcome = await manager.run(
        make_conv(), candidate(), FakePool({}), {},
        session_key="s", parent_request_id="req1",
    )
    assert outcome.outcome == "skipped_no_backend"
    assert outcome.events == candidate()


# ---- unit: prefix stability + logging ----


async def test_reviewer_prompt_extends_executor_prefix(tmp_path):
    reviewer = ScriptedBackend("rev", [script("APPROVE")])
    pool = FakePool({"review": [reviewer]})
    manager = make_manager(tmp_path)
    conv = make_conv()
    await manager.run(conv, candidate(), pool, {}, session_key="s", parent_request_id="r")
    rendered = get_profile("qwen").render(conv, "m")["messages"]
    reviewer_msgs = reviewer.requests[0]["messages"]
    assert reviewer_msgs[: len(rendered)] == rendered
    assert reviewer_msgs[len(rendered)]["role"] == "assistant"
    assert "The answer is 42." in reviewer_msgs[len(rendered)]["content"]
    # tool surface unchanged: the rendered prefix includes the same tools
    assert reviewer.requests[0].get("tools") == get_profile("qwen").render(conv, "m").get("tools")


async def test_rounds_are_logged(tmp_path):
    reviewer = ScriptedBackend("rev", [script(OBJECTION), script("CONCEDE"), script("APPROVE")])
    pool = FakePool({"review": [reviewer]})
    manager = make_manager(tmp_path)
    await manager.run(
        make_conv(), candidate(), pool, {},
        regenerate=revised_regen([]),
        session_key="sess", parent_request_id="req9",
    )
    rows = reviews_records(tmp_path)
    rounds = [r for r in rows if r["kind"] == "debate_round"]
    summary = [r for r in rows if r["kind"] == "debate"]
    assert len(rounds) == 2
    assert rounds[0]["verdict"] == "objection"
    assert rounds[0]["counter"] == "concede"
    assert rounds[1]["verdict"] == "approve"
    assert all(r["parent_request_id"] == "req9" and r["session_key"] == "sess" for r in rows)
    assert summary[0]["outcome"] == "consensus"
    assert summary[0]["rounds"] == 2


# ---- unit: keepalive + codec ----


async def test_keepalive_iter_pings_while_source_is_slow():
    async def slow():
        yield TextDelta("a")
        await asyncio.sleep(0.05)
        yield Done("end_turn")

    events = [e async for e in keepalive_iter(slow(), 0.01)]
    assert any(isinstance(e, Ping) for e in events)
    non_pings = [e for e in events if not isinstance(e, Ping)]
    assert non_pings == [TextDelta("a"), Done("end_turn")]


async def test_codec_handles_ping_events():
    async def gen():
        for ev in [Ping(), TextDelta("x"), Done("end_turn", 1, 1, 0)]:
            yield ev

    pieces = [p async for p in stream_sse(gen(), "m", "id1")]
    assert sum(1 for p in pieces if "event: ping" in p) >= 2
    body = collect([Ping(), TextDelta("x"), Done("end_turn", 1, 1, 0)], "m", "id1")
    assert body["content"] == [{"type": "text", "text": "x"}]


# ---- server wiring ----


def server_settings(tmp_path, mode: str) -> Settings:
    s = Settings()
    s.review.mode = mode
    s.review.reviews_dir = str(tmp_path / "reviews")
    s.log.requests_path = str(tmp_path / "requests.jsonl")
    s.backends = [
        PoolBackendCfg(
            name="b",
            base_url="http://fake/v1",
            model="m",
            roles=["main", "subagent", "fast", "review"],
        )
    ]
    return s


async def _post(app, body: dict):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        return await client.post("/v1/messages", json=body)


def prose_body() -> dict:
    # A question long enough that short-imperative intent never binds a
    # tool-requiring action state; the candidate turn is legitimate prose.
    body = request_body(stream=False)
    body["messages"] = [{
        "role": "user",
        "content": "Please summarize what value the config file assigns in one plain-text sentence.",
    }]
    return body


async def test_shadow_mode_ships_original_and_logs(tmp_path):
    fake = FakeOpenAI()
    fake.push(script("The answer is 42."))  # executor candidate
    fake.push(script("APPROVE"))            # reviewer
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(server_settings(tmp_path, "shadow"), backend_client=backend_client)
    resp = await _post(app, prose_body())
    assert resp.status_code == 200
    assert resp.json()["content"] == [{"type": "text", "text": "The answer is 42."}]
    for _ in range(40):  # debate runs post-response in the background
        if reviews_records(tmp_path):
            break
        await asyncio.sleep(0.05)
    rows = reviews_records(tmp_path)
    assert any(r["kind"] == "debate" and r["outcome"] == "consensus" for r in rows)
    assert len(fake.requests) == 2


async def test_enforce_mode_ships_revised_candidate(tmp_path):
    fake = FakeOpenAI()
    fake.push(script("The answer is 42."))  # executor candidate
    fake.push(script(OBJECTION))            # reviewer round 1
    fake.push(script("CONCEDE"))            # counter-review
    fake.push(script("Revised answer with evidence."))  # regeneration
    fake.push(script("APPROVE"))            # reviewer round 2
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(server_settings(tmp_path, "enforce"), backend_client=backend_client)
    resp = await _post(app, prose_body())
    assert resp.status_code == 200
    assert resp.json()["content"] == [{"type": "text", "text": "Revised answer with evidence."}]
    rows = [
        json.loads(line)
        for line in (tmp_path / "requests.jsonl").read_text().splitlines()
    ]
    main = next(r for r in rows if r.get("debate_outcome"))
    assert main["debate_outcome"] == "consensus"
    assert main["debate_rounds"] == 2
