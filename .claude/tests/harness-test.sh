#!/usr/bin/env bash
# tests/harness-test.sh — the harness's own regression suite.
# Run via ./verify.sh at the repo root. Requires: bash, jq, node, python3, git.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
FAILURES=0
T() { # name, command...
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then echo "PASS $name"; else echo "FAIL $name"; FAILURES=$((FAILURES+1)); fi
}
TY() { # yaml test: SKIP if PyYAML unavailable
  local name="$1"; shift
  if ! python3 -c "import yaml" >/dev/null 2>&1; then echo "SKIP $name (PyYAML not installed)"; return; fi
  T "$name" "$@"
}

# ---- static checks -----------------------------------------------------------
for f in "$HERE"/hooks/*.sh "$HERE"/wf-run.sh "$HERE"/project-template/.claude/verify.sh; do
  T "syntax:$(basename "$f")" bash -n "$f"
done
for j in "$HERE"/settings.json "$HERE"/opencode/opencode.json; do
  T "json:$(basename "$j")" jq empty "$j"
done
for j in "$HERE"/project-template/pyrightconfig.json "$HERE"/project-template/.mcp.json; do
  T "json:template:$(basename "$j")" jq empty "$j"
done
TY "yaml:pre-commit" python3 -c "import yaml,sys; yaml.safe_load(open('$HERE/project-template/.pre-commit-config.yaml'))"
T "template:gitignore-env" grep -q '^\.env' "$HERE/project-template/.gitignore"
T "template:mcp-env-interpolation" bash -c "grep -q '\${EXAMPLE_MCP_TOKEN}' '$HERE/project-template/.mcp.json'"
TY "yaml:ci-workflow" python3 -c "import yaml; yaml.safe_load(open('$HERE/project-template/.github/workflows/ci.yml'))"
T "yaml:harness-ratchet" bash -c "grep -q 'min-level 4' '$HERE/project-template/.github/workflows/harness.yml'"
T "template:ruff-in-pyproject" bash -c "grep -q '\[tool.ruff.format\]' '$HERE/project-template/pyproject.toml' && ! test -f '$HERE/project-template/ruff.toml'"
T "template:settings-project-paths" bash -c "grep -q 'CLAUDE_PROJECT_DIR/.claude/hooks' '$HERE/project-template/.claude/settings.json' && ! grep -q 'HOME' '$HERE/project-template/.claude/settings.json'"
T "template:hooks-committed" bash -c "for h in pretool-guard post-edit-check stop-gate session-context; do test -x '$HERE/project-template/.claude/hooks/'\$h.sh || exit 1; done"
# The template hooks are copies of the root hooks; a fix applied to one and not the
# other ships the bug to every repo that installs the template.
T "template:hooks-parity" bash -c "for h in pretool-guard post-edit-check stop-gate session-context; do cmp -s '$HERE/hooks/'\$h.sh '$HERE/project-template/.claude/hooks/'\$h.sh || exit 1; done"
# macOS has no GNU timeout; every gated command must resolve timeout/gtimeout first.
T "portable:no-bare-timeout" bash -c "! grep -REn 'timeout [0-9]' '$HERE/hooks' '$HERE/project-template/.claude/hooks' '$HERE/wf-run.sh'"
T "session-context:fail-open-detector" bash -c "D=\$(mktemp -d); mkdir -p \$D/.claude; cp '$HERE/hooks/session-context.sh' \$D/; chmod +x \$D/session-context.sh; cd \$D && echo '{\"cwd\":\"'\$D'\"}' | ./session-context.sh | grep -q 'silently ABSENT'"
T "template:commands-exist" bash -c "ls '$HERE/project-template/.claude/commands/'*.md | grep -q ."
T "plugin-import" node --input-type=module -e "import('$HERE/opencode/plugin/harness.js').then(m=>{if(typeof m.Harness!=='function')process.exit(1)})"

# Renders must contain the core sections (drift check)
for doc in "$HERE/skills/workflow/SKILL.md" "$HERE/opencode/agent/orchestrator.md"; do
  for sec in "Feedback protocol" "Task graph" "Small-model discipline" "Ownership and trust" "Self-improvement protocol"; do
    T "render:$(basename "$(dirname "$doc")"):$sec" grep -q "$sec" "$doc"
  done
done

# ---- stop-gate behavior ------------------------------------------------------
WORK=$(mktemp -d); mkdir -p "$WORK/.claude"
printf '#!/usr/bin/env bash\necho FAIL_MARKER\nexit 1\n' > "$WORK/.claude/verify.sh"; chmod +x "$WORK/.claude/verify.sh"
SG_IN() { printf '{"session_id":"%s","cwd":"%s","hook_event_name":"Stop"}' "$1" "$WORK"; }
rm -f "/tmp/claude-stop-gate-hs1"
OUT=$(SG_IN hs1 | "$HERE/hooks/stop-gate.sh")
T "stop-gate:blocks-on-fail" bash -c "printf '%s' '$OUT' | jq -e '.decision == \"block\"'"
T "stop-gate:feedback-written" bash -c "jq -e '.source == \"stop-gate\"' '$WORK/.claude/feedback.jsonl'"
printf '#!/usr/bin/env bash\nexit 0\n' > "$WORK/.claude/verify.sh"
OUT=$(SG_IN hs1 | "$HERE/hooks/stop-gate.sh")
T "stop-gate:silent-on-pass" test -z "$OUT"
printf '#!/usr/bin/env bash\nexit 1\n' > "$WORK/.claude/verify.sh"
rm -f "/tmp/claude-stop-gate-hs2"
for i in 1 2 3; do SG_IN hs2 | CLAUDE_STOP_GATE_MAX=2 "$HERE/hooks/stop-gate.sh" > "$WORK/last.json"; done
T "stop-gate:cap-releases" bash -c "jq -e 'has(\"systemMessage\")' '$WORK/last.json'"

# ---- guard + post-edit -------------------------------------------------------
T "guard:deny-force-push" bash -c "echo '{\"tool_input\":{\"command\":\"git push --force origin main\"}}' | '$HERE/hooks/pretool-guard.sh' | jq -e '.hookSpecificOutput.permissionDecision == \"deny\"'"
T "guard:allow-safe-rm" bash -c "OUT=\$(echo '{\"tool_input\":{\"command\":\"rm -rf build/\"}}' | '$HERE/hooks/pretool-guard.sh'); test -z \"\$OUT\""
printf 'def broken(:' > "$WORK/bad.py"
T "post-edit:blocks-bad-python" bash -c "! (echo '{\"tool_input\":{\"file_path\":\"$WORK/bad.py\"}}' | '$HERE/hooks/post-edit-check.sh')"
printf 'x = 1\n' > "$WORK/good.py"
T "post-edit:passes-good-python" bash -c "echo '{\"tool_input\":{\"file_path\":\"$WORK/good.py\"}}' | '$HERE/hooks/post-edit-check.sh'"

# ---- wf-stats ----------------------------------------------------------------
T "syntax:wf-stats.sh" bash -n "$HERE/wf-stats.sh"
SD=$(mktemp -d); mkdir -p "$SD/.claude/transcripts"; cd "$SD"
echo '{"task":"X1","attempt":1,"outcome":"fail","backend":"cmd","model":"","gate_exit":1,"transcript":"X1-attempt1.log"}' > .claude/transcripts/X1-attempt1.json
echo '{"task":"X1","attempt":2,"outcome":"pass","backend":"cmd","model":"","gate_exit":0,"transcript":"X1-attempt2.log"}' > .claude/transcripts/X1-attempt2.json
echo '{"task":"X2","attempt":1,"outcome":"fail","backend":"cmd","model":"","gate_exit":1,"transcript":"X2-attempt1.log"}' > .claude/transcripts/X2-attempt1.json
echo '{"task":"X2","attempt":2,"outcome":"fail","backend":"cmd","model":"","gate_exit":1,"transcript":"X2-attempt2.log"}' > .claude/transcripts/X2-attempt2.json
T "wf-stats:efficacy-50pct" bash -c "'$HERE/wf-stats.sh' --jsonl | jq -e '.feedback_efficacy.rate == 50'"
T "wf-stats:counts" bash -c "'$HERE/wf-stats.sh' --jsonl | jq -e '.tasks_total == 2 and .tasks_pass == 1 and .attempts_total == 4'"
cd /; rm -rf "$SD"

T "syntax:wf-corpus.sh" bash -n "$HERE/wf-corpus.sh"

# ---- wf-run end-to-end -------------------------------------------------------
WF=$(mktemp -d); mkdir -p "$WF/.claude"; cd "$WF"
git init -q .; git config user.email t@t; git config user.name t; git config commit.gpgsign false
cat > .claude/tasks.json <<'EOF'
{"tasks":[
 {"id":"T1","desc":"create a.txt","verify":"test -f a.txt && ! test -f junk.txt","deps":[],"status":"todo","attempts":0},
 {"id":"T2","desc":"create b.txt","verify":"test -f b.txt","deps":["T1"],"status":"doing","attempts":0},
 {"id":"T3","desc":"impossible","verify":"false","deps":["T2"],"status":"todo","attempts":0},
 {"id":"T4","desc":"behind blocked","verify":"true","deps":["T3"],"status":"todo","attempts":0}
]}
EOF
cat > mock.sh <<'EOF'
#!/usr/bin/env bash
P="$1"; echo "$P" >> prompts.log
case "$P" in
  *"Task T1"*) if grep -q '"task":"T1"' .claude/feedback.jsonl 2>/dev/null; then touch a.txt; else touch junk.txt; fi ;;
  *"Task T2"*) touch b.txt ;;
esac
EOF
chmod +x mock.sh; git add -A; git commit -qm init
"$HERE/wf-run.sh" --backend cmd --agent-cmd ./mock.sh --max 20 >/dev/null 2>&1
RC=$?
T "wf-run:exit-73-blocked" test "$RC" -eq 73
T "wf-run:T1-retried-then-done" bash -c "jq -e '.tasks[] | select(.id==\"T1\") | .status==\"done\" and .attempts==1' .claude/tasks.json"
T "wf-run:crash-sweep-ran-T2" bash -c "grep -q 'Task T2' prompts.log && jq -e '.tasks[] | select(.id==\"T2\") | .status==\"done\"' .claude/tasks.json"
T "wf-run:T3-blocked" bash -c "jq -e '.tasks[] | select(.id==\"T3\") | .status==\"blocked\"' .claude/tasks.json"
T "wf-run:T4-never-dispatched" bash -c "! grep -q 'Task T4' prompts.log"
T "wf-run:rollback-cleaned-junk" test ! -f junk.txt
T "wf-run:transcript-labeled-fail" bash -c "jq -e '.outcome==\"fail\"' .claude/transcripts/T1-attempt1.json"
T "wf-run:transcript-labeled-pass" bash -c "jq -e '.outcome==\"pass\"' .claude/transcripts/T1-attempt2.json"
T "wf-run:pass-committed" bash -c "git log --oneline | grep -q 'wf-run: T1 done'"
T "wf-run:prompt-persisted" test -f .claude/transcripts/T1-attempt1.prompt
"$HERE/wf-corpus.sh" >/dev/null 2>&1
T "corpus:pass-record" bash -c "jq -se 'any(.[]; .meta.task==\"T1\" and .meta.outcome==\"pass\")' .claude/corpus/corpus-pass.jsonl"
T "corpus:repair-detected" bash -c "jq -e '.meta.repair==true' .claude/corpus/corpus-repair.jsonl"
T "corpus:chat-shape" bash -c "jq -e '.messages | map(.role) == [\"user\",\"assistant\"]' .claude/corpus/corpus-pass.jsonl"
echo '{"task":"C1","attempt":1,"outcome":"pass","backend":"claude","model":"claude-opus-4-8","gate_exit":0,"transcript":"C1-attempt1.log"}' > .claude/transcripts/C1-attempt1.json
touch .claude/transcripts/C1-attempt1.log .claude/transcripts/C1-attempt1.prompt
"$HERE/wf-corpus.sh" >/dev/null 2>&1
T "corpus:license-filter" bash -c "! grep -q '\"task\":\"C1\"' .claude/corpus/corpus-pass.jsonl"
cd /; rm -rf "$WORK" "$WF"

echo "----"
if [ "$FAILURES" -gt 0 ]; then echo "harness-test: $FAILURES FAILURE(S)"; exit 1; fi
echo "harness-test: all green"
