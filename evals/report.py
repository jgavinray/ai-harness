#!/usr/bin/env python3
"""Aggregate results.jsonl into a markdown efficacy report."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


FAILURE_CLASSES = ("infra-death", "timeout", "honest-give-up", "wrong-result")


def classify_failure(row: dict) -> str | None:
    """Mechanical failure attribution; None for successful trials."""
    if row.get("success"):
        return None
    if not row.get("input_tokens") and not row.get("output_tokens"):
        return "infra-death"
    if row.get("timed_out"):
        return "timeout"
    if row.get("gave_up_honestly"):
        return "honest-give-up"
    return "wrong-result"


def verdict(success_rate: float) -> str:
    if success_rate >= 0.95:
        return "supported"
    if success_rate >= 0.80:
        return "degraded"
    return "unsupported"


def aggregate_tasks(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    """Group by (model, config, task); per-family failure classes + verdict."""
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["config"], r.get("task", "?"))].append(r)

    out = {}
    for key, rs in groups.items():
        n = len(rs)
        failures: dict[str, int] = {}
        for r in rs:
            cls = classify_failure(r)
            if cls is not None:
                failures[cls] = failures.get(cls, 0) + 1
        rate = sum(bool(r.get("success")) for r in rs) / n
        out[key] = {
            "trials": n,
            "success_rate": rate,
            "verdict": verdict(rate),
            "failures": failures,
            "wall_s_per_session": sum(r.get("session_wall_s", 0) for r in rs) / n,
            "tokens_per_session": sum(r.get("input_tokens", 0) + r.get("output_tokens", 0) for r in rs) / n,
        }
    return out


def markdown_tasks(tasks: dict[tuple[str, str, str], dict]) -> str:
    lines = [
        "## Per-task envelope",
        "",
        "| task | model | config | trials | success | verdict | "
        + " | ".join(FAILURE_CLASSES)
        + " | wall s | tokens |",
        "|" + "---|" * (8 + len(FAILURE_CLASSES)),
    ]
    for (model, config, task), m in sorted(tasks.items(), key=lambda kv: kv[0][2]):
        cells = [
            task, model, config, str(m["trials"]), f"{m['success_rate']:.2f}",
            m["verdict"],
        ]
        cells += [str(m["failures"].get(c, 0)) for c in FAILURE_CLASSES]
        cells += [f"{m['wall_s_per_session']:.0f}", f"{m['tokens_per_session']:.0f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


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
            "review_generated_per_session": sum(r.get("review_generated", 0) for r in rs) / n,
            "contract_feedback_per_session": sum(r.get("contract_feedback", 0) or 0 for r in rs) / n,
            "gave_up_per_session": sum(r.get("gave_up_honestly", 0) or 0 for r in rs) / n,
            "stream_stalls_per_session": sum(r.get("stream_stalls", 0) or 0 for r in rs) / n,
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
            "review_generated_per_session",
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
    rows = load(path)
    report = markdown(aggregate(rows)) + "\n" + markdown_tasks(aggregate_tasks(rows))
    out = path.parent / "report.md"
    out.write_text(report)
    print(report)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
