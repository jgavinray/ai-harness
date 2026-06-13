# Small-LLM Improvement Loop Runbook

Audience: a 14b-class coding model maintaining this repository.

Purpose: run the harness improvement loop, inspect the captured signal, and make
only evidence-backed changes. Do not guess. Do not promote a model, relax a
scaffold, or change routing because it "seems better." Make the same task set
prove it.

## Ground Rules

- Keep user-facing traffic on the current live backends unless a candidate passes
  an eval gate.
- Treat `evals/tasks/*/check.sh` as ground truth for task success.
- Treat one trial as a smoke signal, not proof. Use at least 3 trials before
  promotion or scaffold relaxation.
- Success rate is the first gate. Token and latency wins only matter after
  success is equal or better.
- Never relax a guard or scaffold because it fired often. Frequent firing means
  the model still needs it unless evals prove otherwise.
- Preserve artifacts. Put new eval runs in a dated subdirectory under
  `evals/results/`.

## Important Files

- `harness.toml`: live backend fleet and roles.
- `logs/requests.jsonl`: live per-request metrics.
- `traces/sessions.jsonl`: rendered prompts and emitted events when traces are
  enabled.
- `evals/tasks/`: deterministic eval tasks and checker scripts.
- `evals/run.py`: runs tasks through the harness.
- `evals/report.py`: aggregates result JSONL into a markdown report.
- `scripts/corpus.py`: builds clean SFT corpus rows from successful traces.
- `scripts/shadow_eval.py`: prints eval commands for `roles = ["candidate"]`.
- `scripts/promote_candidate.py`: eval-gated config edit for candidate promotion.
- `scripts/relax_scaffold.py`: eval-gated config edit for retiring scaffolds.

## Captured Signal

Each request log row may contain:

- `success`: eval task checker passed. Present in eval results, not request logs.
- `timed_out`: eval task hit the runner timeout.
- `input_tokens`, `output_tokens`, `cached_tokens`: token cost.
- `wall_ms`, `session_wall_s`: latency.
- `retries`: relay repair attempts.
- `valid_calls`, `invalid_calls`, `repaired_calls`: tool-call quality.
- `tool_surfaced`: hidden tool schema surfaced from catalog.
- `tool_surfaced_names`: names of hidden tools surfaced.
- `guard_fires`: workflow guard counters by guard name.
- `plan_drift`: top-level count of plan drift repairs.
- `memory_tokens`: project-memory prompt cost.
- `research_briefs`: research briefs generated.
- `skill_compiled`: compiled skill procedures injected.
- `capability_fallbacks`: modality fallback count, such as OCR/no-vision fallback.

Interpret these as evidence, not conclusions. A metric tells you where to inspect.
The checker and repeated evals decide whether a change is better.

## Step 1: Confirm The Harness Is Running

```bash
ps -ef | rg 'python -m harness|uvicorn|ai-harness' | rg -v rg
curl -sS http://127.0.0.1:8484/stats
```

Expected:

- one `python -m harness --config harness.toml` process;
- `/stats` returns JSON;
- backend entries show `"down": false`.

If the harness is not running, start it:

```bash
.venv/bin/python -m harness --config harness.toml
```

Use a separate terminal for a long-running harness process.

## Step 2: Run A Baseline/Full Eval Cycle

Use the current main backend unless the task is specifically candidate testing.
For a quick smoke test use `--trials 1`. For decisions use `--trials 3` or more.

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
.venv/bin/python evals/run.py \
  --backend-url http://192.168.0.196:8001/v1 \
  --model qwen3.6-27b \
  --profile qwen \
  --kind vllm \
  --configs baseline,full \
  --trials 3 \
  --out "evals/results/$RUN_ID"

.venv/bin/python evals/report.py "evals/results/$RUN_ID/results.jsonl"
```

Do not edit code while an eval is running. If an eval fails, inspect the result
row and traces before deciding what to change.

## Step 3: Read The Report

Open:

```bash
cat "evals/results/$RUN_ID/report.md"
```

Decision order:

1. Compare `success_rate`.
2. If success is tied, compare `timeout_rate`.
3. If both are tied, compare `tokens_per_session` and `wall_s_per_session`.
4. Use capability counters to identify why behavior changed.

Good signal examples:

- `full.success_rate >= baseline.success_rate` and lower tokens/session.
- `post_repair_invalid_rate` drops after a tool repair change.
- `plan_drift_per_session` drops without a success-rate drop.
- `capability_fallbacks_per_session` appears only when a fallback was expected.

Bad signal examples:

- `full.success_rate < baseline.success_rate`.
- `timeout_rate` increases.
- `invalid_calls` or `retries` increase and success does not improve.
- `guard_fires` drops only because guards were disabled, while success also drops.

## Step 4: Inspect Failures

Find failed rows:

```bash
RUN_ID="20260613-120000"
RUN_ID="$RUN_ID" python3 - <<'PY'
import json
import os
from pathlib import Path
p = Path("evals/results") / os.environ["RUN_ID"] / "results.jsonl"
for row in map(json.loads, p.read_text().splitlines()):
    if not row.get("success"):
        print(row["config"], row["task"], row.get("check_output", "")[-500:])
        print({k: row.get(k) for k in (
            "retries", "invalid_calls", "tool_surfaced",
            "guard_fires", "plan_drift", "timed_out"
        )})
PY
```

Inspect traces for the failed tag:

```bash
RUN_ID="20260613-120000"
FAILED_TAG="qwen3.6-27b-full-fix-test-0"
RUN_ID="$RUN_ID" FAILED_TAG="$FAILED_TAG" python3 - <<'PY'
import json, os
from pathlib import Path
tag = os.environ["FAILED_TAG"]
trace_path = Path("evals/results") / os.environ["RUN_ID"] / "traces/sessions.jsonl"
for line in trace_path.read_text().splitlines():
    row = json.loads(line)
    if row.get("tag") == tag:
        print(row["request_id"], row.get("metrics"))
        for event in row.get("events", []):
            print(event)
PY
```

Use this trace to prove the cause. Example: if `tool_surfaced_names` says
`["Skill"]`, then the model actually attempted hidden `Skill`. If that field is
missing, add observability first; do not guess.

## Step 5: Choose The Adjustment

Make the smallest adjustment that matches the proven failure.

Use this table:

| Signal | First adjustment to investigate |
| --- | --- |
| `invalid_calls` high for a known tool | Improve schema simplification or relay repair for that tool. |
| `tool_surfaced_names` repeatedly names a useful hidden tool | Increase surfacing priority or add a task-specific catalog hint. |
| `tool_surfaced_names` names a distracting meta tool | Add explicit repair guidance or reduce catalog exposure for that class. |
| `guard_fires.verify_after_edit` high and success improves | Keep the guard. The model still needs it. |
| `guard_fires.verify_after_edit` high and repeated evals show no success benefit | Consider `relaxed = ["guard_verify_after_edit"]` for that backend only. |
| `plan_drift` high | Inspect planning status and guard rules; do not disable planning first. |
| `memory_tokens` high with no success gain | Lower `memory.max_chars` or improve distillation selectivity. |
| `research_briefs` high with no success gain | Tighten explicit `research:` detection or cache behavior. |
| candidate success below incumbent | Do not promote. Keep candidate out of live roles. |
| candidate success above incumbent by threshold | Promote with `scripts/promote_candidate.py`. |

## Step 6: Verify A Code Or Config Change

For code changes:

```bash
.venv/bin/python -m pytest tests/ -q
```

Then rerun the failing eval task:

```bash
.venv/bin/python evals/run.py \
  --backend-url http://192.168.0.196:8001/v1 \
  --model qwen3.6-27b \
  --profile qwen \
  --kind vllm \
  --configs full \
  --tasks fix-test \
  --trials 1 \
  --out "evals/results/$RUN_ID-rerun"
```

Only after the narrow rerun passes, run the full config:

```bash
.venv/bin/python evals/run.py \
  --backend-url http://192.168.0.196:8001/v1 \
  --model qwen3.6-27b \
  --profile qwen \
  --kind vllm \
  --configs full \
  --trials 1 \
  --out "evals/results/$RUN_ID-full-check"
```

For decision-making, repeat with `--trials 3` or more.

## Step 7: Promote A Candidate Backend

Candidate backends must have:

```toml
roles = ["candidate"]
```

List candidate eval commands:

```bash
.venv/bin/python scripts/shadow_eval.py --config harness.toml --out evals/results
```

Run the printed commands. Then compare candidate against incumbent:

```bash
.venv/bin/python scripts/promote_candidate.py \
  --results evals/results/candidate-run/results.jsonl \
  --config harness.toml \
  --incumbent qwen3.6-27b \
  --candidate new-model-name \
  --backend-name candidate-backend-name \
  --min-delta 0.05 \
  --roles main
```

Promotion rule:

- If candidate success rate is lower: do not promote.
- If success is equal but token/wall time is better: consider more trials before
  promotion.
- If success exceeds incumbent by `min_delta`: promote.

After promotion, run:

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python evals/run.py \
  --backend-url http://192.168.0.196:8001/v1 \
  --model promoted-model-name \
  --profile qwen \
  --kind vllm \
  --configs full \
  --trials 3 \
  --out evals/results/post-promotion
```

## Step 8: Relax A Scaffold

Relaxation means a specific backend has earned the right to skip a scaffold.
Runtime honors `relaxed = [...]` on that backend.

Allowed scaffold names:

- `workflow_guards`
- `guard_edit_without_read`
- `guard_verify_after_edit`
- `planning` or `planning_scaffold`
- `skills`
- `research`
- `tool_catalog`
- `fewshot`

Example:

```bash
.venv/bin/python scripts/relax_scaffold.py \
  --results evals/results/qwen27-ablation/results.jsonl \
  --config harness.toml \
  --model qwen3.6-27b \
  --backend-name qwen27 \
  --scaffold guard_verify_after_edit \
  --metric plan_drift \
  --max-value 0.0
```

Do not relax from one task or one trial. Use an ablation run:

```bash
.venv/bin/python evals/run.py \
  --backend-url http://192.168.0.196:8001/v1 \
  --model qwen3.6-27b \
  --profile qwen \
  --kind vllm \
  --configs full,ablate-workflow_guards,ablate-planning,ablate-skills \
  --trials 3 \
  --out evals/results/scaffold-ablation
```

Relax only if:

- success does not drop;
- timeout does not rise;
- the target metric stays acceptable;
- token or latency cost improves enough to matter.

## Step 9: Build A Clean Corpus

After a successful eval run:

```bash
.venv/bin/python scripts/corpus.py \
  --traces evals/results/$RUN_ID/traces/sessions.jsonl \
  --results evals/results/$RUN_ID/results.jsonl \
  --out evals/results/$RUN_ID/corpus.jsonl
```

Optional live clean traces:

```bash
.venv/bin/python scripts/corpus.py \
  --traces traces/sessions.jsonl \
  --results evals/results/$RUN_ID/results.jsonl \
  --out evals/results/$RUN_ID/live-corpus.jsonl \
  --include-live
```

Corpus rows are allowed into training only when:

- the eval task passed;
- `invalid_calls == 0`;
- live rows have no retries, invalid calls, degenerate aborts, or loop breaks;
- the conversation is not part of a repeated-call loop.

## Step 10: Train Or Prepare A Candidate

The LoRA script currently emits the training command:

```bash
.venv/bin/python scripts/lora_train.py \
  --corpus evals/results/$RUN_ID/corpus.jsonl \
  --base-model base-model-name \
  --out adapters/candidate-name
```

After training, add the resulting model as a candidate backend in `harness.toml`:

```toml
[[backends]]
name = "candidate-name"
kind = "vllm"
base_url = "http://host:port/v1"
model = "candidate-model"
profile = "qwen"
roles = ["candidate"]
```

Then return to Step 7.

## Step 11: Commit And Record The Evidence

Before committing:

```bash
git diff --check
.venv/bin/python -m pytest tests/ -q
```

In the commit message, say what signal justified the change.

Good commit subject examples:

- `fix: recover from invalid surfaced skill calls`
- `feat: honor eval-relaxed scaffolds`
- `docs: add improvement loop runbook`

Do not include `Co-Authored-By` trailers.

If the change affects eval behavior, keep the report path in the final response:

```text
Eval report: evals/results/20260613-120000/report.md
```

## Stop Conditions

Stop and ask for help when:

- the same failure repeats after two evidence-backed fixes;
- the trace lacks the metric needed to identify cause;
- candidate promotion would reduce success rate;
- a change requires deleting user work or changing live traffic policy;
- backend availability prevents a fair eval comparison.

Before asking, preserve the evidence:

```bash
git status --short
cp evals/results/$RUN_ID/report.md /tmp/harness-report-$RUN_ID.md
```

## Mental Model

The loop is:

```text
live/eval traffic
  -> request logs + traces + checker results
  -> report metrics
  -> evidence-backed code/config adjustment
  -> tests
  -> rerun evals
  -> corpus from clean successful traces
  -> candidate training or backend addition
  -> shadow eval
  -> promote or reject
  -> relax scaffolds only when earned
```

The loop is deterministic because task checkers and numeric thresholds make the
decision. Models may generate candidate behavior, but scripts decide whether it
is better.
