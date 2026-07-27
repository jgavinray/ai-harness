#!/usr/bin/env bash
# PostToolUse (Edit|Write): fast syntax check on the file just written.
# Exit 2 => stderr is fed back to the model immediately (the write already
# happened; this forces a fix on the next step instead of at the stop gate).
set -u

INPUT=$(cat)
FILE=$(printf '%s' "$INPUT" | jq -r '.tool_input.file_path // empty')
{ [ -z "$FILE" ] || [ ! -f "$FILE" ]; } && exit 0

fail() { echo "post-edit-check [$FILE]: $1" >&2; exit 2; }

case "$FILE" in
  *.py)
    ERR=$(python3 -m py_compile "$FILE" 2>&1) || fail "python syntax error: $ERR"
    if command -v ruff >/dev/null 2>&1; then
      ERR=$(ruff check --quiet "$FILE" 2>&1) || fail "ruff: $ERR"
    fi
    ;;
  *.sh|*.bash)
    ERR=$(bash -n "$FILE" 2>&1) || fail "bash syntax error: $ERR"
    ;;
  *.json)
    ERR=$(jq empty "$FILE" 2>&1) || fail "invalid JSON: $ERR"
    ;;
  *.yaml|*.yml)
    if command -v yq >/dev/null 2>&1; then
      ERR=$(yq e '.' "$FILE" 2>&1 >/dev/null) || fail "invalid YAML: $ERR"
    elif command -v python3 >/dev/null 2>&1; then
      ERR=$(python3 -c "import sys,yaml; yaml.safe_load(open(sys.argv[1]))" "$FILE" 2>&1) || fail "invalid YAML: $ERR"
    fi
    ;;
  *.toml)
    if command -v python3 >/dev/null 2>&1; then
      ERR=$(python3 -c "import sys,tomllib; tomllib.load(open(sys.argv[1],'rb'))" "$FILE" 2>&1) || fail "invalid TOML: $ERR"
    fi
    ;;
esac
exit 0
