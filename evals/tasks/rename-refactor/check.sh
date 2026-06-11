#!/usr/bin/env bash
cd "$(dirname "$0")"
if grep -rn "calc_total" --include='*.py' .; then
  echo "old name still present" >&2
  exit 1
fi
grep -qn "def compute_total" billing.py || { echo "compute_total not defined" >&2; exit 1; }
python3 report.py
