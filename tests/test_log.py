import json

from harness.config import Settings
from harness.log import RequestLogger
from harness.relay import run
from harness.profiles.registry import get_profile
from tests.fake_openai import FakeOpenAI, finish_chunk, tool_chunk
from tests.test_backends import make
from tests.test_relay import conv


def test_logger_writes_jsonl(tmp_path):
    logger = RequestLogger(tmp_path / "requests.jsonl")
    logger.write({"request_id": "r1", "wall_ms": 12})
    logger.write({"request_id": "r2", "wall_ms": 30})
    lines = (tmp_path / "requests.jsonl").read_text().strip().split("\n")
    assert [json.loads(l)["request_id"] for l in lines] == ["r1", "r2"]


def test_logger_disabled_when_path_none():
    logger = RequestLogger(None)
    logger.write({"request_id": "r1"})  # no error, no file


async def test_relay_metrics_count_retries():
    fake = FakeOpenAI()
    fake.push([tool_chunk("c1", "Read", '{"nope": 1}'), finish_chunk("tool_calls")])
    fake.push([tool_chunk("c2", "Read", '{"file_path": "/x"}'), finish_chunk("tool_calls")])
    fake.push([finish_chunk("stop")])
    metrics: dict = {}
    [e async for e in run(conv(), get_profile("qwen"), make(fake), Settings(), metrics=metrics)]
    assert metrics["retries"] == 1
    assert metrics["repaired_calls"] == 0
    assert metrics["valid_calls"] == 1


async def test_relay_metrics_repaired():
    fake = FakeOpenAI()
    fake.push([tool_chunk("c1", "Read", '{"file_path": "/x",}'), finish_chunk("tool_calls")])
    metrics: dict = {}
    [e async for e in run(conv(), get_profile("qwen"), make(fake), Settings(), metrics=metrics)]
    assert metrics["repaired_calls"] == 1
    assert metrics["retries"] == 0
