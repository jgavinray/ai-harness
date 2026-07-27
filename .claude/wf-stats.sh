#!/usr/bin/env bash
# wf-stats.sh — aggregate loop telemetry from transcripts + feedback into the
# signals that drive improvement proposals. Deterministic output (sorted, no
# timestamps). Run from a project root; add --jsonl for machine-readable.
set -uo pipefail
T=".claude/transcripts"
FB=".claude/feedback.jsonl"
GRAPH=".claude/tasks.json"
MODE="${1:-text}"

[ -d "$T" ] || { echo "wf-stats: no $T — nothing to report" >&2; exit 0; }
LABELS=$(cat "$T"/*.json 2>/dev/null | jq -s 'sort_by(.task, .attempt)')
[ "$(printf '%s' "$LABELS" | jq 'length')" -gt 0 ] || { echo "wf-stats: no labeled transcripts" >&2; exit 0; }

STATS=$(printf '%s' "$LABELS" | jq --slurpfile fb <(cat "$FB" 2>/dev/null || echo '') '
  . as $l
  | (group_by(.task) | map({
      task: .[0].task,
      attempts: length,
      final: (sort_by(.attempt) | last.outcome),
      models: ([.[].model] | unique | sort)
    }) | sort_by(.task)) as $per_task
  # feedback efficacy: a fail at attempt N followed by a pass at N+1 (same task)
  | ([ $l[] | select(.outcome=="fail") | . as $f
       | ($l[] | select(.task==$f.task and .attempt==($f.attempt+1))) // empty ]) as $followups
  | {
      tasks_total: ($per_task | length),
      tasks_pass: ([$per_task[] | select(.final=="pass")] | length),
      attempts_total: ($l | length),
      attempts_per_pass: (
        ([$per_task[] | select(.final=="pass") | .attempts] | if length>0 then (add/length*100|round)/100 else null end)
      ),
      outcomes: ($l | group_by(.outcome) | map({key: .[0].outcome, value: length}) | from_entries),
      by_model: ($l | group_by(.model) | map({
          model: (.[0].model // ""),
          attempts: length,
          pass: ([.[] | select(.outcome=="pass")] | length)
        }) | sort_by(.model)),
      feedback_efficacy: {
        fails_with_followup: ($followups | length),
        followup_passed: ([$followups[] | select(.outcome=="pass")] | length),
        rate: (if ($followups|length) > 0
               then (([$followups[] | select(.outcome=="pass")] | length) / ($followups|length) * 100 | round)
               else null end)
      },
      flaky_reports: ([ $fb[] | select(.report? // "" | test("FLAKY")) ] | length),
      timeout_reports: ([ $fb[] | select(.exit? == 124) ] | length),
      blocked_tasks: (if "'"$GRAPH"'" | test(".") then [] else [] end),
      per_task: $per_task
    }
')

# blocked list from graph if present
if [ -f "$GRAPH" ]; then
  BLOCKED=$(jq -c '[.tasks[] | select(.status=="blocked") | .id] | sort' "$GRAPH")
  STATS=$(printf '%s' "$STATS" | jq --argjson b "$BLOCKED" '.blocked_tasks = $b')
fi

if [ "$MODE" = "--jsonl" ]; then
  printf '%s\n' "$STATS" | jq -c .
  exit 0
fi

printf '%s' "$STATS" | jq -r '
  "== wf-stats ==",
  "tasks: \(.tasks_pass)/\(.tasks_total) passed | attempts: \(.attempts_total) | attempts-per-pass: \(.attempts_per_pass // "n/a")",
  "outcomes: \(.outcomes | to_entries | map("\(.key)=\(.value)") | join(" "))",
  "feedback efficacy: \(.feedback_efficacy.followup_passed)/\(.feedback_efficacy.fails_with_followup) fail->pass on next attempt (\(.feedback_efficacy.rate // "n/a")%)",
  "flaky: \(.flaky_reports) | timeouts: \(.timeout_reports) | blocked: \(.blocked_tasks | join(",") // "none" | if . == "" then "none" else . end)",
  "by model:",
  (.by_model[] | "  \(if .model == "" then "(default)" else .model end): \(.pass)/\(.attempts) pass"),
  "per task:",
  (.per_task[] | "  \(.task): \(.attempts) attempt(s) -> \(.final)")
'
