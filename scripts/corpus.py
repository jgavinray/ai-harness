#!/usr/bin/env python3
"""Build an outcome-filtered SFT corpus from harness traces + eval results.

  .venv/bin/python scripts/corpus.py \
      --traces traces/sessions.jsonl \
      --results evals/results/results.jsonl \
      --out corpus.jsonl

Keeps only requests from trials that passed their checker, and only requests
whose tool calls all validated (zero invalid_calls). With --include-live,
untagged live-session traces are also kept when execution was mechanically
clean (no invalid calls, no retries, no degenerate aborts). Output: one JSONL record
per request, {"messages": [...full rendered context..., assistant_reply]} in
OpenAI chat format — directly usable for LoRA SFT of a smaller model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from harness.traces import assistant_message  # noqa: E402


def successful_tags(results_path: Path) -> set[str]:
    tags = set()
    for line in results_path.read_text().splitlines():
        row = json.loads(line)
        if row.get("success") and row.get("tag"):
            tags.add(row["tag"])
    return tags


def _clean_execution(metrics: dict) -> bool:
    """A live trace is corpus-grade only if nothing went wrong mechanically.
    retries matter beyond hygiene: a retried request's stored payload no
    longer matches its emitted events (feedback turns were appended)."""
    return not any(
        metrics.get(k) for k in ("invalid_calls", "retries", "degenerate_aborts")
    )


def build(
    traces_path: Path,
    results_path: Path,
    out_path: Path,
    include_live: bool = False,
) -> tuple[int, int]:
    keep_tags = successful_tags(results_path)
    kept = total = 0
    with out_path.open("w") as out:
        for line in traces_path.read_text().splitlines():
            total += 1
            trace = json.loads(line)
            metrics = trace.get("metrics", {})
            tagged_ok = trace.get("tag") in keep_tags
            live_ok = include_live and not trace.get("tag") and _clean_execution(metrics)
            if not (tagged_ok or live_ok):
                continue
            if metrics.get("invalid_calls"):
                continue
            messages = trace["payload"]["messages"] + [assistant_message(trace["events"])]
            record = {"messages": messages}
            if trace["payload"].get("tools"):
                record["tools"] = trace["payload"]["tools"]
            out.write(json.dumps(record, separators=(",", ":")) + "\n")
            kept += 1
    return kept, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="traces/sessions.jsonl")
    ap.add_argument("--results", default="evals/results/results.jsonl")
    ap.add_argument("--out", default="corpus.jsonl")
    ap.add_argument("--include-live", action="store_true",
                    help="also keep untagged live-session traces with clean execution")
    args = ap.parse_args()
    kept, total = build(Path(args.traces), Path(args.results), Path(args.out),
                        include_live=args.include_live)
    print(f"corpus: kept {kept}/{total} requests -> {args.out}")


if __name__ == "__main__":
    main()
