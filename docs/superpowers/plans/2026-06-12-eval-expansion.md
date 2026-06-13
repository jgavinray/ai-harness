# Eval Expansion Implementation Plan

**Status:** done 2026-06-12.

**Goal:** Add the roadmap's first measurement layer after Tool Autonomy: a
tool-discovery task family, a long-horizon drift task family, catalog ablation,
and reporting for the counters future guards/scaffolds need.

## Tasks

- [x] Add `evals/tasks/tool-discovery/`, a task whose prompt explicitly requires a
  non-CORE tool (`WebFetch`) to retrieve the hidden answer from `reference.html`.
- [x] Add `evals/tasks/long-horizon/`, a ten-step normalization task that fails
  from the initial repo state and checks the full end state.
- [x] Add `tool_catalog` to the generated eval config matrix so `full`,
  `baseline`, and `ablate-tool_catalog` measure catalog impact.
- [x] Include `WebFetch` in eval `--allowedTools` so the Claude Code client can
  expose a non-CORE tool for the discovery task.
- [x] Aggregate `tool_surfaced` and `guard_fires` in eval results and render
  per-session values in the markdown report.
- [x] Update eval tests to validate the new task families, catalog ablation, and
  counter reporting.

## Verification

Run:

```bash
.venv/bin/python -m pytest tests/test_evals.py -q
.venv/bin/python -m pytest tests/ -q
```

Observed:

```text
5 passed
162 passed
```
