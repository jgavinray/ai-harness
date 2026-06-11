#!/usr/bin/env bash
# Live smoke test: starts the harness against a real OpenAI-compatible
# backend and exercises one streaming tool-use request.
#
#   OPENAI_BASE_URL=http://localhost:11434/v1 MODEL=qwen2.5-coder:14b ./scripts/smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_URL="${OPENAI_BASE_URL:-http://localhost:11434/v1}"
MODEL="${MODEL:-qwen2.5-coder:14b}"
PROFILE="${PROFILE:-qwen}"
PORT="${PORT:-8484}"

CFG="$(mktemp)"
cat > "$CFG" <<EOF
[server]
port = ${PORT}
[backend]
kind = "openai"
base_url = "${BASE_URL}"
model = "${MODEL}"
[profile]
name = "${PROFILE}"
EOF

.venv/bin/python -m harness --config "$CFG" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true; rm -f "$CFG"' EXIT
sleep 2

OUT="$(curl -sS -N "http://127.0.0.1:${PORT}/v1/messages" \
  -H 'content-type: application/json' \
  -H 'x-api-key: local' \
  -d '{
    "model": "claude-sonnet-4-6",
    "max_tokens": 512,
    "stream": true,
    "system": "You are a coding agent. Use tools when asked.",
    "messages": [{"role": "user", "content": "Read the file /etc/hostname using the Read tool."}],
    "tools": [{
      "name": "Read",
      "description": "Reads a file from the local filesystem.",
      "input_schema": {
        "type": "object",
        "properties": {"file_path": {"type": "string"}},
        "required": ["file_path"]
      }
    }]
  }')"

echo "$OUT"
echo
if grep -q "message_stop" <<<"$OUT"; then
  echo "SMOKE OK: stream completed with message_stop"
else
  echo "SMOKE FAILED: no message_stop in stream" >&2
  exit 1
fi
