#!/usr/bin/env bash
# Runner writes the agent's final reply to answer.txt before calling us.
cd "$(dirname "$0")"
[ -f answer.txt ] || { echo "no answer.txt" >&2; exit 1; }
grep -Eq "DEFECT: *\S*billing\.py:apply_discount" answer.txt
