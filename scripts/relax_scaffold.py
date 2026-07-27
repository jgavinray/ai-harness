#!/usr/bin/env python3
"""Eval-gated scaffold relaxation config edit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def metric_average(results: Path, model: str, metric: str) -> float:
    rows = [
        json.loads(l) for l in results.read_text().splitlines()
        if l.strip() and json.loads(l).get("model") == model
    ]
    if not rows:
        return 0.0
    return sum(float(r.get(metric) or 0.0) for r in rows) / len(rows)


def can_relax(results: Path, model: str, metric: str, max_value: float) -> bool:
    return metric_average(results, model, metric) <= max_value


def relax_config(config: Path, backend_name: str, scaffold: str) -> None:
    lines = config.read_text().splitlines()
    out: list[str] = []
    in_target = False
    wrote = False
    inserted = False
    for line in lines:
        if line.strip() == "[[backends]]":
            if in_target and not wrote:
                out.append(f'relaxed = ["{scaffold}"]')
                inserted = True
            in_target = False
            wrote = False
        if line.strip().startswith("name") and f'"{backend_name}"' in line:
            in_target = True
        if in_target and line.strip().startswith("relaxed"):
            current = _parse_list(line)
            if scaffold not in current:
                current.append(scaffold)
            out.append("relaxed = [" + ", ".join(f'"{x}"' for x in current) + "]")
            wrote = True
            continue
        out.append(line)
    if in_target and not wrote:
        out.append(f'relaxed = ["{scaffold}"]')
        inserted = True
    if not in_target and not inserted and backend_name not in "\n".join(lines):
        raise ValueError(f"backend {backend_name!r} not found")
    config.write_text("\n".join(out) + "\n")


def _parse_list(line: str) -> list[str]:
    raw = line.split("=", 1)[1].strip().strip("[]")
    return [x.strip().strip('"') for x in raw.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend-name", required=True)
    ap.add_argument("--scaffold", required=True)
    ap.add_argument("--metric", default="success")
    ap.add_argument("--max-value", type=float, default=0.0)
    args = ap.parse_args()
    if not can_relax(Path(args.results), args.model, args.metric, args.max_value):
        raise SystemExit("relaxation gate failed")
    relax_config(Path(args.config), args.backend_name, args.scaffold)
    print(f"relaxed {args.scaffold} for {args.backend_name}")


if __name__ == "__main__":
    main()
