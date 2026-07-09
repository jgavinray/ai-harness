#!/usr/bin/env python3
"""Derived DuckDB index over the JSONL data plane. Disposable by design:
the JSONL partitions are the source of truth and this database can be
deleted and rebuilt at any time.

  .venv/bin/python scripts/analytics.py refresh [--db harness.duckdb]
  .venv/bin/python scripts/analytics.py query "SELECT count(*) FROM requests"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _sql_list(files: list[str]) -> str:
    quoted = ", ".join("'" + f.replace("'", "''") + "'" for f in files)
    return f"[{quoted}]"


def _view(con, name: str, files: list[str], select: str) -> None:
    if not files:
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT NULL AS empty WHERE false")
        return
    con.execute(
        f"CREATE OR REPLACE VIEW {name} AS SELECT {select} "
        f"FROM read_json_auto({_sql_list(files)}, format='newline_delimited', "
        "union_by_name=true, ignore_errors=true)"
    )


def refresh(db_path: Path, logs: Path, traces: Path):
    import duckdb

    con = duckdb.connect(str(db_path))
    request_files = sorted(
        str(p)
        for p in [*logs.glob("requests.jsonl*"), *(logs / "requests").glob("*.jsonl")]
    )
    trace_files = sorted(
        str(p) for p in [*traces.glob("sessions*.jsonl"), *traces.glob("*/*.jsonl")]
    )
    _view(con, "requests", request_files, "*")
    # payload/events are heavy; the index keeps the queryable envelope and
    # points back at the JSONL for full records.
    _view(con, "trace_records", trace_files, "ts, tag, session_key, request_id, metrics")
    return con


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["refresh", "query"])
    ap.add_argument("sql", nargs="?", help="SQL for the query command")
    ap.add_argument("--db", default=str(ROOT / "harness.duckdb"))
    ap.add_argument("--logs", default=str(ROOT / "logs"))
    ap.add_argument("--traces", default=str(ROOT / "traces"))
    args = ap.parse_args()

    con = refresh(Path(args.db), Path(args.logs), Path(args.traces))
    if args.command == "refresh":
        n = con.execute("SELECT count(*) FROM requests").fetchone()[0]
        print(f"refreshed {args.db}: requests view sees {n} records")
        return
    if not args.sql:
        sys.exit("query command needs SQL")
    for row in con.execute(args.sql).fetchall():
        print(row)


if __name__ == "__main__":
    main()
