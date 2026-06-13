import json

import httpx

from harness.config import Settings
from harness.server import create_app
from tests.fake_openai import FakeOpenAI, finish_chunk, text_chunk, tool_chunk
from tests.test_server import READ_TOOL, request_body


async def post(settings: Settings, fake: FakeOpenAI, body: dict):
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    settings.backend.base_url = "http://fake/v1"
    app = create_app(settings, backend_client=backend_client)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        return await client.post("/v1/messages", json=body)


async def test_request_logged_with_metrics(tmp_path):
    log = tmp_path / "requests.jsonl"
    settings = Settings()
    settings.log.requests_path = str(log)
    fake = FakeOpenAI()
    fake.push([
        text_chunk("hi"),
        tool_chunk("c1", "Read", '{"file_path": "/x"}'),
        finish_chunk("tool_calls", prompt_tokens=42, completion_tokens=9),
    ])
    resp = await post(settings, fake, request_body(stream=True))
    assert resp.status_code == 200 and "message_stop" in resp.text

    rec = json.loads(log.read_text().strip().split("\n")[0])
    assert rec["input_tokens"] == 42 and rec["output_tokens"] == 9
    assert rec["stop_reason"] == "tool_use"
    assert rec["valid_calls"] == 1 and rec["retries"] == 0
    assert rec["wall_ms"] >= 0 and rec["ttft_ms"] >= 0
    assert rec["request_id"].startswith("msg_")


async def test_no_log_when_disabled(tmp_path):
    settings = Settings()  # log.requests_path defaults to None
    fake = FakeOpenAI()
    fake.push([text_chunk("hi"), finish_chunk("stop")])
    resp = await post(settings, fake, request_body(stream=False))
    assert resp.status_code == 200
    assert list(tmp_path.iterdir()) == []


async def test_request_log_records_memory_tokens(tmp_path):
    log = tmp_path / "requests.jsonl"
    mem = tmp_path / "memory"
    settings = Settings()
    settings.log.requests_path = str(log)
    settings.memory.enabled = True
    settings.memory.dir = str(mem)
    (mem).mkdir()
    (mem / "repo.md").write_text("- use pytest -q\n")
    fake = FakeOpenAI()
    fake.push([text_chunk("hi"), finish_chunk("stop")])
    body = request_body(stream=False)
    body["system"] = "Working directory: /repo"
    resp = await post(settings, fake, body)
    assert resp.status_code == 200
    rec = json.loads(log.read_text().strip())
    assert rec["memory_tokens"] > 0
