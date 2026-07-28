#!/usr/bin/env bash
# wf-run.sh — outer driver for the workflow state machine.
#
# Loop: sweep crashed state -> pick next READY node from the task graph ->
# snapshot -> spawn a FRESH headless agent session seeded with (node, wf-state,
# newest feedback) -> evaluate gates ourselves -> on fail, ROLL BACK the tree so
# the next attempt starts from clean state + feedback -> on pass, commit ->
# repeat until the graph is done or everything ready is blocked.
#
# Contracts:
#   State restoration : every attempt starts from the same snapshot; failed
#                       attempts leave no residue (git reset+clean, .claude
#                       state files preserved across rollback).
#   Context           : disk is memory; each session is fresh and minimal.
#   Ownership         : the DRIVER owns .claude/tasks.json. Agents own
#                       wf-state.json and NOTES.md. feedback.jsonl is
#                       append-only for everyone.
#   Trust             : gates are evaluated by the driver, never the agent.
#   Transcripts       : every session captured + outcome-labeled under
#                       .claude/transcripts/ (teacher-corpus ready).
#
# Graph: .claude/tasks.json
#   {"tasks":[{"id":"T1","desc":"...","verify":"<shell cmd>","deps":["T0"],
#              "status":"todo|doing|done|blocked","attempts":0,
#              "model":"<optional per-node model override>"}]}
#
# Env: WF_BACKEND=claude|opencode|cmd  WF_MAX_ITER=40  WF_NODE_ATTEMPTS=3
#      WF_SESSION_TIMEOUT=1800  WF_MODEL=<default model>  WF_FLAKE_CHECK=0|1
#      WF_AGENT_CMD=<runner for backend cmd>
#      WF_CLAUDE_ARGS / WF_OPENCODE_ARGS = extra args passed through verbatim
#      (e.g. richer transcript output formats).
set -uo pipefail

BACKEND="${WF_BACKEND:-claude}"
MAX_ITER="${WF_MAX_ITER:-40}"
NODE_ATTEMPTS_MAX="${WF_NODE_ATTEMPTS:-3}"
SESSION_TIMEOUT="${WF_SESSION_TIMEOUT:-1800}"
MODEL="${WF_MODEL:-}"
FLAKE_CHECK="${WF_FLAKE_CHECK:-0}"
AGENT_CMD="${WF_AGENT_CMD:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --backend)   BACKEND="$2"; shift 2 ;;
    --max)       MAX_ITER="$2"; shift 2 ;;
    --agent-cmd) AGENT_CMD="$2"; shift 2 ;;
    --model)     MODEL="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done

TIMEOUT_CMD="$(command -v timeout || command -v gtimeout || true)"
GRAPH=".claude/tasks.json"
STATE=".claude/wf-state.json"
FEEDBACK=".claude/feedback.jsonl"
TRANSCRIPTS=".claude/transcripts"
[ -f "$GRAPH" ] || { echo "wf-run: $GRAPH missing — run the planner first" >&2; exit 66; }
mkdir -p .claude "$TRANSCRIPTS"

IS_GIT=0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 && IS_GIT=1
# Automated commits must not depend on the user's signing setup (gpgsign=true with
# no key for the automation identity fails EVERY snapshot -> rollback silently off).
GITC="git"
[ "${WF_GIT_SIGN:-0}" = "1" ] || GITC="git -c commit.gpgsign=false -c tag.gpgsign=false"

# ---- graph ops: driver-owned, serialized via flock when available -----------
graph_write() { # jq-program args...
  local tmp; tmp=$(mktemp)
  if command -v flock >/dev/null 2>&1; then
    ( flock 9; jq "$@" "$GRAPH" > "$tmp" && mv "$tmp" "$GRAPH" ) 9>"$GRAPH.lock"
  else
    jq "$@" "$GRAPH" > "$tmp" && mv "$tmp" "$GRAPH"
  fi
}

next_ready() {
  jq -r '
    .tasks as $t
    | [ .tasks[]
        | select(.status == "todo")
        | select( ([.deps[]?] | all(. as $d | ($t | map(select(.id == $d)) | first | .status) == "done")) ) ]
    | if length == 0 then empty else .[0].id end
  ' "$GRAPH"
}

node_field()   { jq -r --arg id "$1" --arg f "$2" '.tasks[] | select(.id == $id) | .[$f] // empty' "$GRAPH"; }
set_status()   { graph_write --arg id "$1" --arg st "$2" '(.tasks[] | select(.id == $id) | .status) = $st'; }
bump_attempts(){ graph_write --arg id "$1" '(.tasks[] | select(.id == $id) | .attempts) = ((.tasks[] | select(.id == $id) | .attempts // 0) + 1)'; }
get_attempts() { jq -r --arg id "$1" '.tasks[] | select(.id == $id) | .attempts // 0' "$GRAPH"; }

record_feedback() { # id, source, exit, report
  jq -nc --arg task "$1" --arg source "$2" --argjson exit "$3" --arg report "$4" \
    '{source: $source, task: $task, exit: $exit, report: $report}' >> "$FEEDBACK"
}

# ---- snapshot / rollback -----------------------------------------------------
STATE_FILES=("$GRAPH" "$FEEDBACK" "$STATE")

snapshot() { # -> echoes ref, empty if not a git repo
  [ "$IS_GIT" -eq 1 ] || return 0
  git add -A >/dev/null 2>&1
  if ! git diff --cached --quiet 2>/dev/null; then
    $GITC commit -q -m "wf-run: baseline before $1 attempt $2" >/dev/null 2>&1
  fi
  git rev-parse HEAD
}

rollback() { # ref
  [ "$IS_GIT" -eq 1 ] && [ -n "$1" ] || return 0
  local save; save=$(mktemp -d)
  local f
  for f in "${STATE_FILES[@]}"; do [ -f "$f" ] && cp "$f" "$save/$(basename "$f")"; done
  git reset --hard "$1" >/dev/null 2>&1
  git clean -fd -e .claude >/dev/null 2>&1
  for f in "${STATE_FILES[@]}"; do [ -f "$save/$(basename "$f")" ] && cp "$save/$(basename "$f")" "$f"; done
  rm -rf "$save"
}

commit_pass() { # id
  [ "$IS_GIT" -eq 1 ] || return 0
  git add -A >/dev/null 2>&1
  git diff --cached --quiet 2>/dev/null || $GITC commit -q -m "wf-run: $1 done — $(node_field "$1" desc)" >/dev/null 2>&1
}

# ---- gates -------------------------------------------------------------------
eval_gates() { # id -> sets GATE_OK, GATE_REPORT
  local id="$1" node_verify node_out="" node_ok=0 proj_out="" proj_ok=0
  node_verify=$(node_field "$id" verify)
  if [ -n "$node_verify" ]; then
    node_out=$(${TIMEOUT_CMD:+"$TIMEOUT_CMD" 600} bash -c "$node_verify" 2>&1) || node_ok=$?
  fi
  if [ -x .claude/verify.sh ]; then
    proj_out=$(${TIMEOUT_CMD:+"$TIMEOUT_CMD" 600} .claude/verify.sh 2>&1) || proj_ok=$?
  fi
  GATE_OK=$(( node_ok == 0 && proj_ok == 0 ? 0 : 1 ))
  GATE_REPORT=$(printf 'node gate (exit %s):\n%s\nproject gate (exit %s):\n%s' \
    "$node_ok" "$(printf '%s' "$node_out" | tail -c 1500)" \
    "$proj_ok" "$(printf '%s' "$proj_out" | tail -c 1500)")
  GATE_EXIT=$(( node_ok != 0 ? node_ok : proj_ok ))
}

# ---- one agent iteration, fresh context, captured -----------------------------
run_agent() { # prompt, transcript_path
  local prompt="$1" tpath="$2" rc=0
  case "$BACKEND" in
    claude)
      # shellcheck disable=SC2086
      ${TIMEOUT_CMD:+"$TIMEOUT_CMD" "$SESSION_TIMEOUT"} claude -p ${MODEL:+--model "$MODEL"} ${WF_CLAUDE_ARGS:-} "$prompt" >"$tpath" 2>&1 || rc=$? ;;
    opencode)
      # shellcheck disable=SC2086
      ${TIMEOUT_CMD:+"$TIMEOUT_CMD" "$SESSION_TIMEOUT"} opencode run ${MODEL:+--model "$MODEL"} ${WF_OPENCODE_ARGS:-} "$prompt" >"$tpath" 2>&1 || rc=$? ;;
    cmd)
      [ -n "$AGENT_CMD" ] || { echo "wf-run: --agent-cmd required for backend=cmd" >&2; exit 64; }
      ${TIMEOUT_CMD:+"$TIMEOUT_CMD" "$SESSION_TIMEOUT"} "$AGENT_CMD" "$prompt" >"$tpath" 2>&1 || rc=$? ;;
    *) echo "wf-run: unknown backend $BACKEND" >&2; exit 64 ;;
  esac
  return $rc
}

label_transcript() { # id, attempt, outcome, tpath
  jq -nc --arg task "$1" --argjson attempt "$2" --arg outcome "$3" \
        --arg backend "$BACKEND" --arg model "${NODE_MODEL:-$MODEL}" \
        --argjson gate_exit "${GATE_EXIT:-0}" --arg transcript "$(basename "$4")" \
    '{task: $task, attempt: $attempt, outcome: $outcome, backend: $backend, model: $model, gate_exit: $gate_exit, transcript: $transcript}' \
    > "${4%.log}.json"
}

build_prompt() { # id
  local id="$1" desc verify fb=""
  desc=$(node_field "$id" desc)
  verify=$(node_field "$id" verify)
  [ -f "$FEEDBACK" ] && fb=$(tail -1 "$FEEDBACK")
  printf 'Execute exactly one task via the workflow state machine, then stop.\nTask %s: %s\nAcceptance (node gate): %s\nWorkflow state: %s\nLatest feedback (address the FIRST item, quote it, fix it, re-check): %s\nRules: never edit .claude/tasks.json (driver-owned). Write state transitions to .claude/wf-state.json. Do not start any other task. Do not claim completion — the driver verifies.\n' \
    "$id" "$desc" "$verify" "$(cat "$STATE" 2>/dev/null || echo '{}')" "${fb:-none}"
}

# ---- crash recovery: nodes stuck in "doing" from a dead run -------------------
STUCK=$(jq -r '[.tasks[] | select(.status=="doing") | .id] | join(" ")' "$GRAPH")
if [ -n "$STUCK" ]; then
  echo "wf-run: sweeping crashed nodes back to todo: $STUCK"
  for s in $STUCK; do set_status "$s" todo; done
fi

# ---- main loop ---------------------------------------------------------------
ITER=0
while [ "$ITER" -lt "$MAX_ITER" ]; do
  ITER=$((ITER + 1))
  ID=$(next_ready)
  if [ -z "$ID" ]; then
    TODO=$(jq '[.tasks[] | select(.status=="todo")] | length' "$GRAPH")
    [ "$TODO" -gt 0 ] && echo "wf-run: $TODO task(s) unreachable behind blocked dependencies" >&2
    break
  fi

  ATTEMPT=$(( $(get_attempts "$ID") + 1 ))
  NODE_MODEL=$(node_field "$ID" model)
  SAVED_MODEL="$MODEL"; [ -n "$NODE_MODEL" ] && MODEL="$NODE_MODEL"

  set_status "$ID" doing
  REF=$(snapshot "$ID" "$ATTEMPT")
  TPATH="$TRANSCRIPTS/${ID}-attempt${ATTEMPT}.log"
  echo "wf-run[$ITER/$MAX_ITER]: $ID (attempt $ATTEMPT/$NODE_ATTEMPTS_MAX)${NODE_MODEL:+ model=$NODE_MODEL}"

  PROMPT=$(build_prompt "$ID")
  printf '%s' "$PROMPT" > "${TPATH%.log}.prompt"
  AGENT_RC=0
  run_agent "$PROMPT" "$TPATH" || AGENT_RC=$?
  MODEL="$SAVED_MODEL"

  if [ "$AGENT_RC" -eq 124 ]; then
    GATE_EXIT=124
    label_transcript "$ID" "$ATTEMPT" timeout "$TPATH"
    record_feedback "$ID" "wf-run" 124 "session timeout after ${SESSION_TIMEOUT}s — decompose the task or investigate the hang"
    rollback "$REF"
    bump_attempts "$ID"
    if [ "$(get_attempts "$ID")" -ge "$NODE_ATTEMPTS_MAX" ]; then set_status "$ID" blocked; else set_status "$ID" todo; fi
    continue
  fi

  eval_gates "$ID"

  if [ "$GATE_OK" -eq 0 ] && [ "$FLAKE_CHECK" = "1" ]; then
    FIRST_REPORT="$GATE_REPORT"
    eval_gates "$ID"
    if [ "$GATE_OK" -ne 0 ]; then
      GATE_REPORT=$(printf 'FLAKY VERIFICATION: gates passed then failed on immediate re-run. Fix the nondeterminism before the task can pass.\nsecond run:\n%s' "$GATE_REPORT")
    fi
  fi

  if [ "$GATE_OK" -eq 0 ]; then
    label_transcript "$ID" "$ATTEMPT" pass "$TPATH"
    set_status "$ID" done
    rm -f "$STATE"
    commit_pass "$ID"
    echo "wf-run: $ID done"
    continue
  fi

  label_transcript "$ID" "$ATTEMPT" fail "$TPATH"
  record_feedback "$ID" "wf-run" "$GATE_EXIT" "$GATE_REPORT"
  rollback "$REF"
  bump_attempts "$ID"
  if [ "$(get_attempts "$ID")" -ge "$NODE_ATTEMPTS_MAX" ]; then
    set_status "$ID" blocked
    rm -f "$STATE"
    echo "wf-run: $ID BLOCKED after $NODE_ATTEMPTS_MAX attempts" >&2
  else
    set_status "$ID" todo
  fi
done

# ---- summary (deterministic order) ------------------------------------------
DONE=$(jq '[.tasks[] | select(.status=="done")] | length' "$GRAPH")
BLOCKED=$(jq -c '[.tasks[] | select(.status=="blocked") | .id] | sort' "$GRAPH")
TOTAL=$(jq '.tasks | length' "$GRAPH")
echo "wf-run: $DONE/$TOTAL done, blocked: $BLOCKED, iterations: $ITER, transcripts: $TRANSCRIPTS/"
[ "$BLOCKED" = "[]" ] && [ "$DONE" -eq "$TOTAL" ] && exit 0
[ "$ITER" -ge "$MAX_ITER" ] && { echo "wf-run: iteration cap hit" >&2; exit 74; }
exit 73
