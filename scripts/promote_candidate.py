#!/usr/bin/env python3
"""Promote eval-gated candidate backends by editing their role list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def success_rate(results: Path, model: str) -> float:
    rows = [json.loads(l) for l in results.read_text().splitlines() if l.strip()]
    selected = [r for r in rows if r.get("model") == model]
    if not selected:
        return 0.0
    return sum(bool(r.get("success")) for r in selected) / len(selected)


def should_promote(results: Path, incumbent: str, candidate: str, min_delta: float) -> bool:
    return success_rate(results, candidate) >= success_rate(results, incumbent) + min_delta


def promote_config(config: Path, backend_name: str, roles: list[str]) -> None:
    lines = config.read_text().splitlines()
    in_block = False
    out = []
    replaced = False
    for line in lines:
        if line.strip() == "[[backends]]":
            in_block = True
        elif in_block and line.startswith("[") and line.strip() != "[[backends]]":
            in_block = False
        if in_block and line.strip().startswith("name") and f'"{backend_name}"' not in line:
            in_block = False
        if in_block and line.strip().startswith("roles"):
            out.append("roles = [" + ", ".join(f'"{r}"' for r in roles) + "]")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        raise ValueError(f"backend {backend_name!r} roles not found")
    config.write_text("\n".join(out) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--backend-name", required=True)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--roles", default="main")
    args = ap.parse_args()
    if not should_promote(Path(args.results), args.incumbent, args.candidate, args.min_delta):
        raise SystemExit("candidate did not pass promotion gate")
    promote_config(Path(args.config), args.backend_name, args.roles.split(","))
    print(f"promoted {args.backend_name} to roles {args.roles}")


if __name__ == "__main__":
    main()
