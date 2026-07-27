#!/usr/bin/env bash
# SessionStart: plain stdout from this event is injected as context.
# Gives the model ground truth at turn zero so it doesn't rediscover state.
set -u
cd "$(jq -r '.cwd // "."')" 2>/dev/null || exit 0

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "git branch: $(git branch --show-current 2>/dev/null)"
  DIRTY=$(git status --porcelain 2>/dev/null | head -20)
  [ -n "$DIRTY" ] && printf 'uncommitted changes:\n%s\n' "$DIRTY"
  echo "last commit: $(git log -1 --oneline 2>/dev/null)"
fi

if [ -f TASKS.md ]; then
  echo "--- TASKS.md (first 40 lines) ---"
  head -40 TASKS.md
fi

if [ -f .claude/wf-state.json ]; then
  echo "--- workflow state (resume here) ---"
  jq -c . .claude/wf-state.json 2>/dev/null || cat .claude/wf-state.json
fi

if [ -f .claude/feedback.jsonl ]; then
  echo "--- newest gate feedback ---"
  tail -1 .claude/feedback.jsonl
fi

if [ -x .claude/verify.sh ]; then
  echo "verifier present: .claude/verify.sh (Stop gate is ACTIVE — you cannot finish until it exits 0)"
fi
exit 0
