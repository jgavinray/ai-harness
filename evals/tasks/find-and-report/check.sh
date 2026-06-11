#!/usr/bin/env bash
# Runner writes the agent's final reply to answer.txt before calling us.
cd "$(dirname "$0")"
[ -f answer.txt ] || { echo "no answer.txt" >&2; exit 1; }
grep -q "settings\.py" answer.txt && grep -Eq "(^|[^0-9])9([^0-9]|$)" answer.txt
