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
    config.write_text(_rewrite_roles(config.read_text(), backend_name, roles))


def _rewrite_roles(text: str, backend_name: str, roles: list[str]) -> str:
    in_block = False
    out = []
    replaced = False
    for line in text.splitlines():
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
    return "\n".join(out) + "\n"


def propose_config(config: Path, backend_name: str, roles: list[str]) -> str:
    """Write <config>.proposed and return the unified diff; never touches the
    live config (promotion stays a human-reviewed commit until the loop has a
    track record — spec, umbrella principle 1)."""
    import difflib

    old = config.read_text()
    new = _rewrite_roles(old, backend_name, roles)
    proposed = config.with_suffix(config.suffix + ".proposed")
    proposed.write_text(new)
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=str(config), tofile=str(proposed),
    ))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--backend-name", required=True)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--roles", default="main")
    ap.add_argument("--apply", action="store_true",
                    help="edit the config in place instead of proposing a diff")
    args = ap.parse_args()
    if not should_promote(Path(args.results), args.incumbent, args.candidate, args.min_delta):
        raise SystemExit("candidate did not pass promotion gate")
    if args.apply:
        promote_config(Path(args.config), args.backend_name, args.roles.split(","))
        print(f"promoted {args.backend_name} to roles {args.roles}")
    else:
        print(propose_config(Path(args.config), args.backend_name, args.roles.split(",")))


if __name__ == "__main__":
    main()
