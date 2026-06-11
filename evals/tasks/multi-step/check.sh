#!/usr/bin/env bash
cd "$(dirname "$0")"
# test file must be untouched
if [ -n "$(git diff --name-only HEAD -- test_pipeline.py 2>/dev/null)" ]; then
  echo "test_pipeline.py was modified" >&2
  exit 1
fi
python3 test_pipeline.py
