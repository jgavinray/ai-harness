import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("duckdb")

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import analytics


def test_refresh_indexes_partitions_and_legacy_files(tmp_path):
    requests = tmp_path / "logs" / "requests"
    requests.mkdir(parents=True)
    (requests / "2026-07-08.jsonl").write_text('{"backend":"q","input_tokens":5}\n')
    (requests / "2026-07-09.jsonl").write_text('{"backend":"q","input_tokens":7}\n')
    (tmp_path / "logs" / "requests.jsonl").write_text('{"backend":"q","input_tokens":1}\n')
    day = tmp_path / "traces" / "2026-07-09"
    day.mkdir(parents=True)
    (day / "abcdef.jsonl").write_text(json.dumps({
        "ts": 1.0, "tag": "", "session_key": "abcdef", "request_id": "r1",
        "payload": {"messages": []}, "events": [], "metrics": {"retries": 0},
    }) + "\n")

    con = analytics.refresh(tmp_path / "h.duckdb", tmp_path / "logs", tmp_path / "traces")
    assert con.execute("SELECT count(*) FROM requests").fetchone()[0] == 3
    assert con.execute("SELECT sum(input_tokens) FROM requests").fetchone()[0] == 13
    assert con.execute("SELECT count(*) FROM trace_records").fetchone()[0] == 1


def test_refresh_with_no_data_yields_empty_views(tmp_path):
    con = analytics.refresh(tmp_path / "h.duckdb", tmp_path / "logs", tmp_path / "traces")
    assert con.execute("SELECT count(*) FROM requests").fetchone()[0] == 0
