#!/usr/bin/env bash
# wf-corpus.sh — build SFT training corpora from labeled wf-run transcripts.
#
# Emits chat-format JSONL (axolotl / LLaMA-Factory / torchtune compatible):
#   {"messages":[{"role":"user","content":<prompt>},
#                {"role":"assistant","content":<transcript>}],
#    "meta":{"task","attempt","outcome","backend","model","repair"}}
#
# Two sets:
#   corpus-pass.jsonl   — gate-passed trajectories (rejection-sampled positives)
#   corpus-repair.jsonl — the subset of passes whose prompt carried feedback
#                         from a prior failed attempt (fail->pass repair signal;
#                         teaches feedback-following, the scarcest small-model skill)
#
# License hygiene (always on, no override flag): records with backend "claude"
# or a model matching /claude|anthropic/i are excluded from BOTH sets — those
# transcripts remain available to stats/retro (rules improvement), not training.
# Additional exclusions: WF_CORPUS_EXCLUDE_MODELS (extended regex).
#
# Usage: wf-corpus.sh [output-dir]   (default .claude/corpus)
set -uo pipefail
T=".claude/transcripts"
OUT="${1:-.claude/corpus}"
EXTRA_EXCLUDE="${WF_CORPUS_EXCLUDE_MODELS:-}"

[ -d "$T" ] || { echo "wf-corpus: no $T" >&2; exit 0; }
mkdir -p "$OUT"
: > "$OUT/corpus-pass.jsonl"
: > "$OUT/corpus-repair.jsonl"

INCLUDED=0; EXCLUDED_LICENSE=0; SKIPPED_INCOMPLETE=0; REPAIR=0

# Deterministic order: sorted label filenames
for LBL in $(ls "$T"/*.json 2>/dev/null | sort); do
  BASE="${LBL%.json}"
  LOG="$BASE.log"; PROMPT_F="$BASE.prompt"

  OUTCOME=$(jq -r '.outcome' "$LBL")
  [ "$OUTCOME" = "pass" ] || continue

  if [ ! -f "$LOG" ] || [ ! -f "$PROMPT_F" ]; then
    SKIPPED_INCOMPLETE=$((SKIPPED_INCOMPLETE + 1))
    continue
  fi

  BACKEND=$(jq -r '.backend // ""' "$LBL")
  MODEL=$(jq -r '.model // ""' "$LBL")
  if [ "$BACKEND" = "claude" ] || printf '%s' "$MODEL" | grep -qiE 'claude|anthropic'; then
    EXCLUDED_LICENSE=$((EXCLUDED_LICENSE + 1))
    continue
  fi
  if [ -n "$EXTRA_EXCLUDE" ] && printf '%s' "$MODEL" | grep -qE "$EXTRA_EXCLUDE"; then
    EXCLUDED_LICENSE=$((EXCLUDED_LICENSE + 1))
    continue
  fi

  # repair pair: this passing attempt's prompt carried real prior feedback
  IS_REPAIR=false
  grep -q 'Latest feedback' "$PROMPT_F" && ! grep -q 'Latest feedback (address the FIRST item, quote it, fix it, re-check): none' "$PROMPT_F" \
    && IS_REPAIR=true

  REC=$(jq -nc \
    --rawfile prompt "$PROMPT_F" \
    --rawfile output "$LOG" \
    --slurpfile meta "$LBL" \
    --argjson repair "$IS_REPAIR" \
    '{messages: [{role:"user", content:$prompt}, {role:"assistant", content:$output}],
      meta: ($meta[0] | {task, attempt, outcome, backend, model} + {repair: $repair})}')

  printf '%s\n' "$REC" >> "$OUT/corpus-pass.jsonl"
  INCLUDED=$((INCLUDED + 1))
  if [ "$IS_REPAIR" = "true" ]; then
    printf '%s\n' "$REC" >> "$OUT/corpus-repair.jsonl"
    REPAIR=$((REPAIR + 1))
  fi
done

echo "wf-corpus: $INCLUDED pass records ($REPAIR repair), $EXCLUDED_LICENSE excluded by license filter, $SKIPPED_INCOMPLETE skipped (missing prompt/log) -> $OUT/"
