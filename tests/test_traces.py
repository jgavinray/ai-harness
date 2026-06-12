import json
import sys
from pathlib import Path

import httpx

from harness.config import Settings
from harness.ir import Done, TextDelta, ToolCall
from harness.server import create_app
from harness.traces import TraceStore, assistant_message, serialize_event
from tests.fake_openai import FakeOpenAI, finish_chunk, text_chunk, tool_chunk
from tests.test_server import request_body

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import corpus  # noqa: E402


def test_event_round_trip_to_assistant_message():
    events = [
        serialize_event(TextDelta("let me ")),
        serialize_event(TextDelta("read it")),
        serialize_event(ToolCall("t1", "Read", {"file_path": "/x"})),
        serialize_event(Done("tool_use", 10, 5)),
    ]
    msg = assistant_message(events)
    assert msg["role"] == "assistant"
    assert msg["content"] == "let me read it"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"file_path": "/x"}


def test_trace_store_tag_and_write(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_TRACE_TAG", "model-full-fix-test-0")
    store = TraceStore(tmp_path)
    store.append("sess1", "req1", {"messages": []}, [TextDelta("x")], {"retries": 0})
    rec = json.loads((tmp_path / "sessions.jsonl").read_text())
    assert rec["tag"] == "model-full-fix-test-0"
    assert rec["events"] == [{"t": "text", "text": "x"}]


async def test_server_writes_traces(tmp_path):
    settings = Settings()
    settings.backend.base_url = "http://fake/v1"
    settings.traces.enabled = True
    settings.traces.dir = str(tmp_path)
    fake = FakeOpenAI()
    fake.push([
        text_chunk("on it"),
        tool_chunk("c1", "Read", '{"file_path": "/x"}'),
        finish_chunk("tool_calls"),
    ])
    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake.app), base_url="http://fake"
    )
    app = create_app(settings, backend_client=backend_client)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        await client.post("/v1/messages", json=request_body(stream=False))
    rec = json.loads((tmp_path / "sessions.jsonl").read_text())
    assert rec["payload"]["messages"][0]["role"] == "system"
    assert any(e["t"] == "tool_call" for e in rec["events"])
    assert rec["metrics"]["valid_calls"] == 1


def test_corpus_filters_by_outcome_and_validity(tmp_path):
    traces = tmp_path / "sessions.jsonl"
    results = tmp_path / "results.jsonl"
    out = tmp_path / "corpus.jsonl"
    base = {
        "payload": {"messages": [{"role": "user", "content": "hi"}],
                    "tools": [{"type": "function", "function": {"name": "Read"}}]},
        "events": [{"t": "text", "text": "ok"}, {"t": "done", "stop_reason": "end_turn",
                                                  "input_tokens": 1, "output_tokens": 1,
                                                  "cached_tokens": 0}],
    }
    traces.write_text("\n".join(json.dumps({**base, "tag": tag, "metrics": m}) for tag, m in [
        ("good-0", {"invalid_calls": 0}),
        ("bad-0", {"invalid_calls": 0}),       # trial failed -> excluded
        ("good-1", {"invalid_calls": 2}),      # invalid calls -> excluded
    ]))
    results.write_text("\n".join(json.dumps(r) for r in [
        {"tag": "good-0", "success": True},
        {"tag": "good-1", "success": True},
        {"tag": "bad-0", "success": False},
    ]))
    kept, total = corpus.build(traces, results, out)
    assert (kept, total) == (1, 3)
    rec = json.loads(out.read_text())
    assert rec["messages"][-1]["role"] == "assistant"
    assert rec["tools"][0]["function"]["name"] == "Read"


def test_corpus_includes_clean_live_traces(tmp_path):
    # Live (untagged) traffic has no eval result row; with include_live it
    # must be kept when execution was clean, and a clean execution also
    # requires retries == 0 (a retried request's stored payload no longer
    # matches its emitted events).
    traces = tmp_path / "sessions.jsonl"
    results = tmp_path / "results.jsonl"
    out = tmp_path / "corpus.jsonl"
    base = {
        "payload": {"messages": [{"role": "user", "content": "hi"}]},
        "events": [{"t": "text", "text": "ok"}, {"t": "done", "stop_reason": "end_turn",
                                                  "input_tokens": 1, "output_tokens": 1,
                                                  "cached_tokens": 0}],
    }
    traces.write_text("\n".join(json.dumps({**base, "tag": "", "metrics": m}) for m in [
        {"invalid_calls": 0, "retries": 0, "degenerate_aborts": 0},   # kept
        {"invalid_calls": 0, "retries": 1, "degenerate_aborts": 0},   # retried -> excluded
        {"invalid_calls": 1, "retries": 0, "degenerate_aborts": 0},   # invalid -> excluded
        {"invalid_calls": 0, "retries": 0, "degenerate_aborts": 1},   # degenerate -> excluded
    ]))
    results.write_text("")
    kept, total = corpus.build(traces, results, out, include_live=True)
    assert (kept, total) == (1, 4)
    # default behavior unchanged: untagged traces stay excluded
    kept, total = corpus.build(traces, results, out)
    assert (kept, total) == (0, 4)


def test_corpus_excludes_loopy_sessions(tmp_path):
    # A session that repeats an identical tool call >= 3 times is behavioral
    # garbage even when every request is mechanically clean; none of its
    # rows may enter the corpus.
    traces = tmp_path / "sessions.jsonl"
    results = tmp_path / "results.jsonl"
    out = tmp_path / "corpus.jsonl"
    clean = {"invalid_calls": 0, "retries": 0, "degenerate_aborts": 0}

    def row(session_user, call_args):
        return {
            "tag": "", "metrics": clean,
            "payload": {"messages": [{"role": "system", "content": "s"},
                                     {"role": "user", "content": session_user}]},
            "events": [{"t": "tool_call", "id": "c", "name": "Bash", "arguments": call_args},
                       {"t": "done", "stop_reason": "tool_use",
                        "input_tokens": 1, "output_tokens": 1, "cached_tokens": 0}],
        }

    loopy = [row("loop session", {"command": "git worktree list"}) for _ in range(4)]
    healthy = [row("good session", {"command": f"ls /{i}"}) for i in range(4)]
    traces.write_text("\n".join(json.dumps(r) for r in loopy + healthy))
    results.write_text("")
    kept, total = corpus.build(traces, results, out, include_live=True)
    assert (kept, total) == (4, 8)
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert all("good session" in json.dumps(r) for r in recs)
