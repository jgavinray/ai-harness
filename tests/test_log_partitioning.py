import json
import time

from harness.log import RequestLogger
from harness.server import _request_log_paths, _stats_state_path


def test_request_logger_directory_mode_writes_dated_partition(tmp_path):
    logger = RequestLogger(None, directory=tmp_path / "requests")
    logger.write({"a": 1})
    logger.write({"a": 2})
    day = time.strftime("%Y-%m-%d")
    f = tmp_path / "requests" / f"{day}.jsonl"
    assert f.exists()
    rows = [json.loads(l) for l in f.read_text().splitlines()]
    assert [r["a"] for r in rows] == [1, 2]


def test_single_file_mode_unchanged(tmp_path):
    path = tmp_path / "requests.jsonl"
    logger = RequestLogger(path)
    logger.write({"a": 1})
    assert json.loads(path.read_text())["a"] == 1


def test_request_log_paths_accepts_directory(tmp_path):
    d = tmp_path / "requests"
    d.mkdir()
    (d / "2026-07-08.jsonl").write_text("{}\n")
    (d / "2026-07-09.jsonl").write_text("{}\n")
    assert [p.name for p in _request_log_paths(d)] == [
        "2026-07-08.jsonl",
        "2026-07-09.jsonl",
    ]


def test_stats_state_path_for_directory(tmp_path):
    d = tmp_path / "requests"
    d.mkdir()
    assert _stats_state_path(d) == d / "stats_state.json"
