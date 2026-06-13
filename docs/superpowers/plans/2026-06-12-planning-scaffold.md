# Planning Scaffold Implementation Plan

**Status:** done 2026-06-13.

**Goal:** Add the platform spec's planning scaffold: a plan-role backend produces
one structured step list per main session, and the executor receives a compact
plan/status scaffold at a stable system-prompt position.

## Tasks

- [x] Add `[planning]` config with opt-in enablement, max step count, and prompt
  character budget.
- [x] Add `PlanningManager`, keyed by session hash, to generate a plan once and
  cache it for subsequent turns.
- [x] Dispatch planning to a `plan` role backend when available, falling back to
  `main` and then any live backend.
- [x] Inject an `## Execution plan` block at the end of the system prompt with a
  compact `Plan status: Step N/M` line.
- [x] Keep planning off the user-facing path on backend failure; log `plan_error`
  and continue without wedging the request.
- [x] Wire `plan_drift` into request metrics and eval reporting so future drift
  detectors have a stable counter.
- [x] Add server tests proving the plan is generated once, cached, and injected
  into executor prompts.

## Verification

Run:

```bash
.venv/bin/python -m pytest tests/test_server.py::test_planning_scaffold_generated_once_and_injected tests/test_evals.py -q
.venv/bin/python -m pytest tests/ -q
```

Observed:

```text
6 passed
167 passed
```
