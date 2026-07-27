#!/usr/bin/env bash
# Stop hook: refuse to let the model stop until the project verifier passes.
# Mechanism: on Stop, run .claude/verify.sh; on nonzero exit, emit
# {"decision":"block","reason":...} which forces the turn to continue with the
# failure output injected as feedback. Exit-0 JSON is required — JSON on any
# other exit code is ignored by Claude Code.
#
# Loop guard: per-session counter, capped at CLAUDE_STOP_GATE_MAX (default 8).
set -u

INPUT=$(cat)
SESSION_ID=$(printf '%s' "$INPUT" | jq -r '.session_id // "unknown"')
CWD=$(printf '%s' "$INPUT" | jq -r '.cwd // "."')

MAX_ITER="${CLAUDE_STOP_GATE_MAX:-8}"
COUNT_FILE="${TMPDIR:-/tmp}/claude-stop-gate-${SESSION_ID}"

# Resolve the project verifier. No verifier => nothing to gate.
VERIFY=""
if [ -x "$CWD/.claude/verify.sh" ]; then
  VERIFY="$CWD/.claude/verify.sh"
elif [ -x "$CWD/verify.sh" ]; then
  VERIFY="$CWD/verify.sh"
fi
[ -z "$VERIFY" ] && exit 0

COUNT=0
[ -f "$COUNT_FILE" ] && COUNT=$(cat "$COUNT_FILE" 2>/dev/null || echo 0)

if [ "$COUNT" -ge "$MAX_ITER" ]; then
  rm -f "$COUNT_FILE"
  jq -n --arg msg "stop-gate: iteration cap ($MAX_ITER) reached; releasing stop. Verification may still be failing — inspect manually." \
    '{systemMessage: $msg}'
  exit 0
fi

OUTPUT=$(cd "$CWD" && timeout 600 "$VERIFY" 2>&1)
STATUS=$?

if [ "$STATUS" -eq 0 ]; then
  rm -f "$COUNT_FILE"
  exit 0
fi

echo $((COUNT + 1)) > "$COUNT_FILE"
TAIL=$(printf '%s' "$OUTPUT" | tail -c 4000)

# Durable feedback record for the workflow state machine (.claude/feedback.jsonl).
# Deterministic fields only: no timestamps.
mkdir -p "$CWD/.claude" 2>/dev/null
jq -n --arg session "$SESSION_ID" --arg tail "$TAIL" --argjson attempt $((COUNT + 1)) --argjson exit "$STATUS" \
  '{source: "stop-gate", session: $session, attempt: $attempt, exit: $exit, report: $tail}' \
  >> "$CWD/.claude/feedback.jsonl" 2>/dev/null

jq -n \
  --arg reason "verify.sh FAILED (exit $STATUS, attempt $((COUNT + 1))/$MAX_ITER). You are not done. Fix the first failure below, then re-run .claude/verify.sh yourself before finishing. Output tail:
$TAIL" \
  '{decision: "block", reason: $reason}'
exit 0
