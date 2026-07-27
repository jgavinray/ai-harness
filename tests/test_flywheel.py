import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from harness.config import Settings
from harness.flywheel import (
    next_nightly,
    nightly_jobs,
    prune_partitions,
    run_job,
    sentinel_verdicts,
    write_sentinel_state,
)
from harness.log import RequestLogger


def test_next_nightly_rolls_to_tomorrow_when_past():
    now = datetime(2026, 7, 9, 4, 30, tzinfo=UTC)
    nxt = next_nightly(now, hour=3)
    assert (nxt.day, nxt.hour, nxt.minute) == (10, 3, 0)
    early = datetime(2026, 7, 9, 1, 0, tzinfo=UTC)
    assert next_nightly(early, hour=3).day == 9


def test_prune_partitions_removes_only_dated_entries(tmp_path):
    requests = tmp_path / "requests"
    requests.mkdir()
    (requests / "2026-01-01.jsonl").write_text("{}\n")
    (requests / "2026-07-09.jsonl").write_text("{}\n")
    (requests / "legacy-requests.jsonl").write_text("{}\n")
    traces = tmp_path / "traces"
    (traces / "2026-01-01").mkdir(parents=True)
    (traces / "2026-01-01" / "s.jsonl").write_text("{}\n")
    (traces / "2026-07-09").mkdir()
    (traces / "sessions.jsonl").write_text("{}\n")

    now = datetime(2026, 7, 9, tzinfo=UTC).timestamp()
    removed = prune_partitions(requests, traces, days=90, now=now)
    assert (requests / "2026-01-01.jsonl").exists() is False
    assert (requests / "2026-07-09.jsonl").exists()
    assert (requests / "legacy-requests.jsonl").exists()
    assert (traces / "2026-01-01").exists() is False
    assert (traces / "2026-07-09").exists()
    assert (traces / "sessions.jsonl").exists()
    assert len(removed) == 2


def test_run_job_records_outcome(tmp_path):
    logger = RequestLogger(tmp_path / "flywheel.jsonl")
    rec = run_job("hello", [sys.executable, "-c", "print('hi')"], Path.cwd(), logger)
    assert rec["rc"] == 0
    assert "hi" in rec["output_tail"]
    logged = json.loads((tmp_path / "flywheel.jsonl").read_text())
    assert logged["job"] == "hello"
    bad = run_job("boom", [sys.executable, "-c", "raise SystemExit(3)"], Path.cwd(), logger)
    assert bad["rc"] == 3


def test_sentinel_verdicts_and_state(tmp_path):
    rows = [
        {"model": "m", "config": "full", "task": "fix-test", "success": True,
         "input_tokens": 1, "output_tokens": 1, "session_wall_s": 1},
        {"model": "m", "config": "full", "task": "rename-refactor", "success": False,
         "input_tokens": 1, "output_tokens": 1, "session_wall_s": 1},
    ]
    results = tmp_path / "results.jsonl"
    results.write_text("\n".join(json.dumps(r) for r in rows))
    verdicts = sentinel_verdicts(results)
    assert verdicts["fix-test"] == "supported"
    assert verdicts["rename-refactor"] == "unsupported"
    state = write_sentinel_state(tmp_path / "sentinel.json", verdicts)
    assert state["degraded"] == ["rename-refactor"]
    assert json.loads((tmp_path / "sentinel.json").read_text())["degraded"] == ["rename-refactor"]


def test_flywheel_status_reads_jobs_and_sentinel(tmp_path):
    from harness.server import _flywheel_status
    s = Settings()
    s.flywheel.log_path = str(tmp_path / "flywheel.jsonl")
    s.flywheel.sentinel_state_path = str(tmp_path / "sentinel.json")
    assert _flywheel_status(s) == {}
    (tmp_path / "flywheel.jsonl").write_text(
        json.dumps({"ts": 1.0, "job": "corpus", "rc": 0, "duration_s": 2.0}) + "\n"
        + json.dumps({"ts": 2.0, "job": "corpus", "rc": 1, "duration_s": 3.0}) + "\n"
    )
    (tmp_path / "sentinel.json").write_text(json.dumps({"ts": 3.0, "degraded": ["x"]}))
    status = _flywheel_status(s)
    assert status["jobs"]["corpus"]["rc"] == 1  # latest record wins
    assert status["sentinel_degraded"] is True


def test_nightly_jobs_follow_settings(tmp_path):
    s = Settings()
    s.traces.dir = "traces"
    s.traces.enabled = True
    s.traces.layout = "partitioned"
    s.memory.enabled = True
    jobs = dict(nightly_jobs(s))
    assert "memory_distill" in jobs
    assert "--traces" in jobs["corpus"] and "traces" in jobs["corpus"]
    assert "--include-live" in jobs["corpus"]
    assert any("analytics.py" in a for a in jobs["analytics"])
    assert any("gate_health.py" in a for a in jobs["gate_health"])
    s.review.mode = "shadow"
    jobs = dict(nightly_jobs(s))
    assert any("review_patterns.py" in a for a in jobs["review_patterns"])
    assert s.review.reviews_dir in jobs["review_patterns"]
    s.review.mode = "off"
    s.memory.enabled = False
    s.traces.layout = "sessions"
    jobs = dict(nightly_jobs(s))
    assert "memory_distill" not in jobs
    assert "gate_health" not in jobs
    assert "review_patterns" not in jobs
    assert "traces/sessions.jsonl" in jobs["corpus"]


def test_training_due_fires_on_corpus_growth(tmp_path):
    # Phase 2 (flywheel spec): when the gated corpus grows past the
    # threshold, the flywheel emits a training_due record with the prepared
    # commands. Training itself stays human-triggered: the RTX Pro 6000 is
    # occupied by serving, so the job must never grab the GPU on its own.
    from harness.flywheel import check_training_due
    s = Settings()
    s.flywheel.train_threshold_rows = 100
    s.flywheel.corpus_path = str(tmp_path / "corpus.jsonl")
    s.flywheel.train_state_path = str(tmp_path / "train_state.json")
    logger = RequestLogger(tmp_path / "flywheel.jsonl")
    (tmp_path / "corpus.jsonl").write_text("{}\n" * 40)
    check_training_due(s, logger)
    assert not (tmp_path / "flywheel.jsonl").exists()  # below threshold

    (tmp_path / "corpus.jsonl").write_text("{}\n" * 150)
    check_training_due(s, logger)
    rec = json.loads((tmp_path / "flywheel.jsonl").read_text().splitlines()[-1])
    assert rec["job"] == "training_due"
    assert rec["corpus_rows"] == 150
    assert any("qlora_train.py" in c for c in rec["commands"])

    # growth resets: no re-fire until another threshold's worth accrues
    check_training_due(s, logger)
    lines = (tmp_path / "flywheel.jsonl").read_text().splitlines()
    assert len(lines) == 1


def test_training_due_disabled_by_default(tmp_path):
    from harness.flywheel import check_training_due
    s = Settings()
    s.flywheel.corpus_path = str(tmp_path / "corpus.jsonl")
    s.flywheel.train_state_path = str(tmp_path / "train_state.json")
    (tmp_path / "corpus.jsonl").write_text("{}\n" * 100000)
    logger = RequestLogger(tmp_path / "flywheel.jsonl")
    check_training_due(s, logger)
    assert not (tmp_path / "flywheel.jsonl").exists()
