#!/usr/bin/env python3
"""Nightly review-pattern report: mine one day of adversarial-debate logs.

Spec 2026-07-19 (adversarial review loop): the debate is a sensor. Recurring
objection shapes are the candidates for future deterministic guard rules, so
this job groups one day of logs/reviews/<date>.jsonl objections by a
normalized signature (quotes masked, digits stripped) and writes
logs/review_patterns/<date>.json. Report only: never touches serving,
always exits 0 on readable input, and a missing day is not an error.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import time
from pathlib import Path

_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"|`[^`]*`")
_DIGITS = re.compile(r"\d+")
_ITEM_SPLIT = re.compile(r"\n(?=\s*\d+\.\s)")


def signature(text: str) -> str:
    masked = _QUOTED.sub("…", text)
    masked = _DIGITS.sub("", masked)
    return " ".join(masked.lower().replace(".", " ").split())[:160]


def _items(objection: str) -> list[str]:
    parts = [p.strip() for p in _ITEM_SPLIT.split(objection)]
    return [p for p in parts if p]


def scan_day(path: Path) -> dict:
    rounds = 0
    debates = 0
    outcomes: collections.Counter = collections.Counter()
    counts: collections.Counter = collections.Counter()
    examples: dict[str, str] = {}
    for line in path.open():
        if not line.strip():
            continue
        record = json.loads(line)
        kind = record.get("kind")
        if kind == "debate":
            debates += 1
            outcomes[record.get("outcome") or "unknown"] += 1
        elif kind == "debate_round":
            rounds += 1
            for item in _items(record.get("objection") or ""):
                sig = signature(item)
                if not sig:
                    continue
                counts[sig] += 1
                examples.setdefault(sig, item[:300])
    patterns = [
        {"signature": sig, "count": n, "example": examples[sig]}
        for sig, n in counts.most_common()
    ]
    return {
        "rounds": rounds,
        "debates": debates,
        "outcomes": dict(outcomes),
        "patterns": patterns,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    args = parser.parse_args(argv)
    day_file = Path(args.reviews_dir) / f"{args.date}.jsonl"
    if not day_file.exists():
        print(f"no reviews for {args.date}; nothing to do")
        return 0
    report = scan_day(day_file)
    report["date"] = args.date
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.date}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"{args.date}: {report['debates']} debates, {report['rounds']} rounds, "
        f"{len(report['patterns'])} objection patterns -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
