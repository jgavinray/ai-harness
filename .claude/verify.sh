#!/usr/bin/env bash
# Harness self-gate: same contract as any project — exit 0 = done.
set -euo pipefail
exec "$(dirname "$0")/tests/harness-test.sh"
