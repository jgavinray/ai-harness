#!/usr/bin/env python3
"""Nightly gate-health report: mine one day of traces for enforcement events.

Spec 2026-07-11 (default-open enforcement): telemetry is part of the gate.
Every denial the relay emits is a bracketed marker in the trace text events;
this job counts them per gate per tool and writes
logs/gate_health/<date>.json. A gate that fires often on sessions that still
end fine is a gate under suspicion — the next allowlist-vs-user fight should
surface here the following morning, not in the owner's session weeks later.
Report only: never touches serving, always exits 0 on readable input.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from datetime import date
from pathlib import Path

DENIAL_PATTERNS = {
    "action_state": re.compile(r"\[action state denied (\w+)"),
    "preflight": re.compile(r"\[preflight denied (\w+)"),
    "invalid_call": re.compile(r"\[invalid tool call (\w+)"),
    "harness_failure": re.compile(r"\[harness\] (Task step failed)"),
}


def _text_events(record: dict) -> str:
    parts = []
    for event in record.get("events") or []:
        if isinstance(event, str):
            try:
                event = json.loads(event)
            except ValueError:
                continue
        if isinstance(event, dict) and event.get("t") == "text":
            parts.append(event.get("text", ""))
    return "".join(parts)


def scan_day(day_dir: Path) -> dict:
    denials: collections.Counter = collections.Counter()
    sessions = 0
    sessions_with_denials = 0
    for trace in sorted(day_dir.glob("*.jsonl")):
        sessions += 1
        hit = False
        for line in trace.open():
            if not line.strip():
                continue
            text = _text_events(json.loads(line))
            for gate, pattern in DENIAL_PATTERNS.items():
                for match in pattern.findall(text):
                    denials[f"{gate}:{match}"] += 1
                    hit = True
        if hit:
            sessions_with_denials += 1
    return {
        "date": day_dir.name,
        "sessions": sessions,
        "sessions_with_denials": sessions_with_denials,
        "denials": dict(denials),
    }


def write_report(day_dir: Path, out_dir: Path) -> Path:
    report = scan_day(day_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{day_dir.name}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir", required=True)
    ap.add_argument("--date", default=date.today().isoformat())  # noqa: DTZ011
    ap.add_argument("--out-dir", default="logs/gate_health")
    args = ap.parse_args()
    day_dir = Path(args.traces_dir) / args.date
    if not day_dir.is_dir():
        print(f"gate_health: no traces for {args.date}; nothing to report")
        return
    out = write_report(day_dir, Path(args.out_dir))
    report = json.loads(out.read_text())
    print(
        f"gate_health {report['date']}: {report['sessions']} sessions, "
        f"{report['sessions_with_denials']} with denials, "
        f"denials={report['denials'] or '{}'}"
    )


if __name__ == "__main__":
    main()
