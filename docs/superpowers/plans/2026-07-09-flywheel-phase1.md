# Flywheel Phase 1: Compose-Native Automation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Checkbox steps.

**Goal:** `docker compose up -d` yields serving + a `flywheel` service that runs the deterministic loops nightly (memory distill, corpus rebuild, retention, DuckDB refresh, skill compile) and the envelope sentinel weekly (full n=20 eval; any family dropping below `supported` raises a flag surfaced in `/stats`). Zero host cron.

**Architecture:** `src/harness/flywheel.py` — one asyncio scheduler process (`python -m harness.flywheel`), jobs run as subprocesses of the bundled scripts, every run appends one JSONL record via `RequestLogger` (rotation for free) to `logs/flywheel.jsonl`; sentinel verdicts reuse `evals/report.aggregate_tasks`, state lands in `logs/flywheel_sentinel.json`; `/stats` gains a `flywheel` section reading both. Image gains git+node+claude CLI and the `scripts/`+`evals/` trees.

## Global Constraints
- TDD; `.venv/bin/pytest -q` green per task; commits per task with the standard trailer.
- Scheduler math and job-runner are pure functions, unit-tested without docker.
- Serving container is untouched by flywheel failures (separate process, shared volumes only).
- Fix the Phase-0 bug: memory volume must mount at `/home/harness/.ai-harness` (container user is `harness`, not root).

### Task 1: FlywheelCfg + scheduler/jobs module (TDD)
`[flywheel]` config: `enabled=False`, `nightly_hour=3`, `sentinel_weekday=6` (-1 disables), `sentinel_trials=20`, `retention_days=90`, `corpus_path="corpus/corpus.jsonl"`, `log_path="logs/flywheel.jsonl"`, `sentinel_state_path="logs/flywheel_sentinel.json"`.
`harness/flywheel.py` pure pieces: `next_nightly(now, hour)`, `prune_partitions(requests_dir, traces_dir, days, now)` (date-named files/dirs only; legacy names untouched), `run_job(name, argv, cwd, logger)` (subprocess, duration, rc, output tail → one record), `sentinel_verdicts(results_path)` (via evals/report), `write_sentinel_state(path, degraded)`. Async `main()` loop: nightly = distill → corpus rebuild → retention → analytics refresh → compile skills; sentinel day adds the envelope run. `python -m harness.flywheel --config …` entrypoint.

### Task 2: script compat fixes (TDD where logic changes)
- `memory_distill.py --traces` accepts partitioned dir (same `_trace_lines` pattern as corpus).
- `compile_skills.py --skills-dir` default `~/.claude/skills`.
- `evals/run.py` PYTHON falls back to `sys.executable` when `.venv` is absent (container).
- `corpus.py successful_tags` tolerates a missing results file (live-only corpus).

### Task 3: /stats flywheel section (TDD)
`_flywheel_status(settings)`: last record per job from the tail of `flywheel.jsonl` + sentinel state file; `{}` when disabled/absent. Wire into `/stats`.

### Task 4: image + compose + config; deploy and verify
Dockerfile: add git, nodejs/npm, `npm i -g @anthropic-ai/claude-code`; COPY `scripts/` and `evals/`; create `/home/harness/.ai-harness` and `/app/corpus` owned by harness.
Compose: fix memory volume target to `/home/harness/.ai-harness` (both services); new `flywheel` service (same image, flywheel command, shared volumes incl. `./corpus` and `./evals/results`).
harness.toml: `[flywheel] enabled = true` (+ non-default cadences only).
Verify: both containers healthy; force one manual nightly cycle (`docker compose exec flywheel python -m harness.flywheel --once nightly`) and check `logs/flywheel.jsonl` records + corpus file; `/stats` shows the flywheel section. Sentinel verified via `--once sentinel --trials 2` smoke (full n=20 runs on schedule).
