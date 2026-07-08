# Brick 2: Runaway-Turn Backstop + Ruler v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Checkbox steps.

**Goal:** No session can zombie until the client timeout (the last observed failure class), and the eval report gains per-task failure attribution + envelope verdicts so every later brick is measurable.

**Evidence:** Both 300s hangs in brick-1 verification (`multi-step-2` run 1, `multi-step-4` run 2) are the only two requests with `first_attempt_constraints = 1`; the stream goes silent at the thinking→tool-call transition and never finishes. Coherent thinking, no loop. The repair-path constraint (older) does not show this.

**Architecture:** (a) `pipeline.first_attempt_constraints` config flag, default False — the feature failed 2/2 live firings; off until an eval proves a working variant (umbrella principle 1 in reverse). (b) `pipeline.stream_idle_timeout_s` (default 120, above worst-case re-prefill of 20–60 s; 0 disables): a wrapper around the parse stream yields a stall sentinel when no event arrives in time; the relay converts it to an honest `[harness]` failure with accumulated usage, metric `stream_stalls`. (c) `evals/report.py` gains per-task rows, failure classes (infra-death = zero tokens, timeout, wrong-result = failed check with real work), and per-family verdicts (supported ≥0.95 / degraded ≥0.80 / unsupported); `evals/run.py` aggregates the new counters.

### Task 1: Gate first-attempt constraints (default off)
- Test: vLLM backend + tool-required state → no `guided_json` in request, metric 0; with `first_attempt_constraints=True` → `guided_json` present, metric 1.
- Implement: config field + `and settings.pipeline.first_attempt_constraints` at relay.py:324.

### Task 2: Stream idle timeout → honest failure
- Test: FakeOpenAI script gains `{"_stall_ms": N}` (async sleep, emits nothing); with `stream_idle_timeout_s=0.2` and a 1500 ms stall, relay yields `[harness]`-prefixed TextDelta + `Done("end_turn")` with accumulated usage; `metrics["stream_stalls"] == 1`.
- Implement: `_iter_with_idle_timeout` async wrapper + `_STALLED` sentinel + stall dispatch before other feedback handling; `import asyncio`.

### Task 3: Ruler v2
- Test: `evals/report.py` aggregation unit test — rows with mixed outcomes produce per-task failure classes and correct verdicts.
- Implement: per-(model, config, task) table + verdict column; `aggregate_log` adds `contract_feedback`, `gave_up_honestly`, `stream_stalls`, `action_state_blocks`.

### Task 4: Full-suite run + envelope report
- `pytest -q` green, commit per task.
- Launch: all 7 task families, `full` config, `--trials 20`, background; then 35B bake-off if 192.168.0.33:8000 answers. Produce `docs/reports/` envelope report from ruler v2.
