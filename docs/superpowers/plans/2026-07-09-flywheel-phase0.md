# Flywheel Phase 0: Clean Slate + Data Plane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Checkbox steps.

**Goal:** Other-project cruft evicted from source and repo; request/trace exhaust becomes date-partitioned JSONL with a derived DuckDB index; live compose deployment updated (memory volume added so the learning layer survives recreates).

**Architecture:** `pipeline.path_aliases` config replaces dev-pr constants; `RequestLogger` gains directory mode (`logs/requests/YYYY-MM-DD.jsonl`); `TraceStore` gains `layout = "partitioned"` (`traces/YYYY-MM-DD/<session>.jsonl`); single-file modes stay for the eval runner. `scripts/analytics.py` builds a disposable `harness.duckdb` over the partitions (optional dep, never imported by serving code).

## Global Constraints

- `.venv/bin/pytest -q` green after every task; TDD per task.
- Never stage `docs/nvidia-gpu-fan-control.md` deletions — it gets *committed* (fleet infra runbook).
- Deployment changes only via docker-compose (rebuild + `up -d`).
- Commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

### Task 1: Purge cruft
- `git rm docker/harness.toml` (orphaned by compose mount change); `rm -rf recovery/`; delete `evals/results/final-audit*` (gitignored June debugging runs; July campaign dirs stay — reports cite them); `git add docs/nvidia-gpu-fan-control.md` (commit as infra runbook).

### Task 2: path aliases become config
- `PipelineCfg.path_aliases: list[list[str]] = []`.
- `guards.py`: delete `BAD_DEV_PR_PREFIX`/`GOOD_DEV_PR_PREFIX`; `normalize_confused_paths(call, aliases)`; preflight passes `settings.pipeline.path_aliases`.
- `pipeline/path_canon.py`: aliases from settings, not module constant; `canonicalize_text(text, aliases)`.
- Update all call sites/tests (`test_path_canon`, `test_relay` dev-pr test, `test_system_prompt`, `test_critic`, `test_server`) to configure `path_aliases = [["/Users/jgavinray/dev-pr", "/Users/jgavinray/dev/pr"]]` explicitly — capability tested, hardcode gone. Grep `dev-pr` in `src/` must return nothing.

### Task 3: skills dir default
- `SkillsCfg.dir` and `evals/configs.py`: `~/.codex/skills` → `~/.claude/skills`. Test asserts the default.

### Task 4: partitioned request logs
- `LogCfg.requests_dir: str | None = None`. `RequestLogger(path, directory=None)`: directory mode writes `<dir>/<YYYY-MM-DD>.jsonl` (date at write time, size rotation still applies per file).
- `server.py`: construct with both; `_stats_state_path` and `_seed_stats` handle directory mode (glob `<dir>/*.jsonl` sorted).
- Tests: directory mode writes dated file; seed-stats reads partitions.

### Task 5: partitioned traces
- `TracesCfg.layout: Literal["sessions","partitioned"] = "sessions"`. Partitioned append → `<dir>/<YYYY-MM-DD>/<session_key[:16] or "untagged">.jsonl`.
- `scripts/corpus.py --traces` accepts a file or a directory (rglob `*.jsonl`).
- Tests: partitioned layout paths; corpus reads a directory.

### Task 6: DuckDB derived index
- `pyproject` optional extra `analytics = ["duckdb"]`; install into `.venv`.
- `scripts/analytics.py`: `refresh` (re)creates `harness.duckdb` with views `requests` (over `logs/requests.jsonl*` + `logs/requests/*.jsonl`) and `trace_records` (over trace partitions + legacy `sessions.jsonl`), `query "<SQL>"` convenience. Rebuildable: deleting the db is always safe.
- Test (importorskip duckdb): build over tmp partitions, `SELECT count(*)` matches.

### Task 7: deploy
- `harness.toml`: `[log] requests_dir = "logs/requests"` (drop `requests_path`), `[traces] layout = "partitioned"`.
- `docker-compose.yml`: named volume `harness-memory:/root/.ai-harness` (learning layer survives recreates — pulled forward from Phase 1 because we're recreating anyway).
- Rebuild, `up -d`, verify: healthy, stats seeded from legacy + new logs, smoke request lands in `logs/requests/<today>.jsonl`, trace lands in `traces/<today>/`.
