#!/usr/bin/env bash
# Project verifier — the single definition of "done".
# Exit 0 = the Stop gate releases. Anything else = the loop continues.
# Keep every check deterministic: pinned toolchains, no network flakiness,
# no time-dependent assertions.
set -euo pipefail
cd "$(dirname "$0")/.."

# --- add real checks; delete the placeholder block below ---
# Rust:    cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test --quiet
# Python:  ruff check . && ruff format --check . && python -m pytest -q
# Node:    npm run lint && npm test
# Go:      gofmt -l . | (! grep .) && go vet ./... && go test ./...

echo "verify.sh: no checks configured — add project checks, then this gate becomes active" >&2
exit 0
