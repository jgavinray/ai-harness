#!/usr/bin/env python3
"""Aggregate results.jsonl into a markdown efficacy report."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def aggregate(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Group by (model, config); compute the efficacy metrics."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["config"])].append(r)

    out = {}
    for key, rs in groups.items():
        n = len(rs)
        calls = sum(r.get("valid_calls", 0) + r.get("invalid_calls", 0) for r in rs)
        malformed = sum(r.get("repaired_calls", 0) + r.get("invalid_calls", 0) + r.get("retries", 0) for r in rs)
        out[key] = {
            "trials": n,
            "success_rate": sum(bool(r.get("success")) for r in rs) / n,
            "timeout_rate": sum(bool(r.get("timed_out")) for r in rs) / n,
            "malformed_call_rate": malformed / calls if calls else 0.0,
            "post_repair_invalid_rate": sum(r.get("invalid_calls", 0) for r in rs) / calls if calls else 0.0,
            "retries_per_session": sum(r.get("retries", 0) for r in rs) / n,
            "tool_surfaced_per_session": sum(r.get("tool_surfaced", 0) for r in rs) / n,
            "guard_fires_per_session": sum(r.get("guard_fires", 0) for r in rs) / n,
            "plan_drift_per_session": sum(r.get("plan_drift", 0) for r in rs) / n,
            "capability_fallbacks_per_session": sum(r.get("capability_fallbacks", 0) for r in rs) / n,
            "research_briefs_per_session": sum(r.get("research_briefs", 0) for r in rs) / n,
            "skill_compiled_per_session": sum(r.get("skill_compiled", 0) for r in rs) / n,
            "memory_tokens_per_session": sum(r.get("memory_tokens", 0) for r in rs) / n,
            "tokens_per_session": sum(r.get("input_tokens", 0) + r.get("output_tokens", 0) for r in rs) / n,
            "wall_s_per_session": sum(r.get("session_wall_s", 0) for r in rs) / n,
        }
    return out


def markdown(agg: dict[tuple[str, str], dict]) -> str:
    cols = ["trials", "success_rate", "timeout_rate", "malformed_call_rate",
            "post_repair_invalid_rate", "retries_per_session",
            "tool_surfaced_per_session", "guard_fires_per_session",
            "plan_drift_per_session",
            "capability_fallbacks_per_session",
            "research_briefs_per_session",
            "skill_compiled_per_session",
            "memory_tokens_per_session",
            "tokens_per_session", "wall_s_per_session"]
    lines = [
        "# Efficacy report",
        "",
        "| model | config | " + " | ".join(c.replace("_", " ") for c in cols) + " |",
        "|" + "---|" * (len(cols) + 2),
    ]
    for (model, config), m in sorted(agg.items()):
        cells = []
        for c in cols:
            v = m[c]
            cells.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        lines.append(f"| {model} | {config} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "evals/results/results.jsonl")
    report = markdown(aggregate(load(path)))
    out = path.parent / "report.md"
    out.write_text(report)
    print(report)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
