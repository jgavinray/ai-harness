# Self-Improving Flywheel

**Date:** 2026-07-09
**Status:** Approved direction. Sub-project of the umbrella spec
(`2026-06-12-small-llm-platform-design.md`), building on the consistency spec
(140/140 envelope, report `2026-07-08-envelope-27b.md`). Four phased bricks;
each gets its own plan and lands behind the eval gate.

## Problem

The runtime loop is certified and deployed, but the system is static: every
growth-loop station exists as a script a human must remember to run, the
trainer targets the wrong hardware for the main fleet, the data exhaust is
two unbounded flat JSONL files, and other-project cruft (dev-pr path rewrites
hardcoded in `guards.py`, codex-era skill paths, orphaned configs) pollutes
the core.

## Goal

The harness improves without an operator: exhaust accumulates in a queryable
data plane, deterministic loops run on schedule inside the compose
deployment, fine-tuned candidate adapters are trained on gated live data,
shadow-evaled, and promoted by config diff — and a weekly envelope sentinel
catches any regression before the user does.

## Constraints (inherited, non-negotiable)

- JSONL artifacts are the source of truth; any index is derived and
  disposable. No new always-on services besides compose containers.
- 14b-maintainable: plain Python, small files, tests as the safety net.
- Nothing ships without an eval delta; candidates never serve live traffic
  (`candidate` role, already router-enforced).
- **Deployment is docker-compose only.** All automation runs as compose
  services sharing the serving image and volumes — no host cron, no systemd.

## Architecture

```
                 ┌─ compose: ai-harness (serving, unchanged role) ─┐
 Claude Code ──▶ │ certified pipeline → backends                   │
                 │ exhaust: logs/requests/YYYY-MM-DD.jsonl         │
                 │          traces/YYYY-MM-DD/<session>.jsonl      │
                 └─────────────────────────────────────────────────┘
                                   │ shared volumes
                 ┌─ compose: flywheel (same image, scheduler cmd) ─┐
                 │ nightly: memory_distill · corpus append ·       │
                 │          partition retention · duckdb refresh   │
                 │ weekly:  envelope sentinel (n=20 × families)    │
                 │ on-threshold: lora train → shadow eval →        │
                 │               promotion proposal                │
                 │ every job → run-report JSONL + /stats surface   │
                 └─────────────────────────────────────────────────┘
```

## Phases

### Phase 0 — clean slate + data plane

1. **Purge cruft:** delete `recovery/`, orphaned `docker/harness.toml`;
   archive superseded eval-result dirs (gitignored disk hygiene); commit the
   GPU-fan runbook under `docs/` (fleet infra, belongs in-repo).
2. **Path aliases become config:** `pipeline.path_aliases` (list of
   `[bad, good]` pairs, default empty) replaces the hardcoded dev-pr
   constants in `guards.py`/`path_canon.py`. Capability preserved,
   other-project data evicted from source.
3. **Skills dir default** `~/.codex/skills` → `~/.claude/skills` (code and
   eval configs).
4. **Partitioned exhaust:** `[log] requests_dir` writes
   `logs/requests/YYYY-MM-DD.jsonl`; `[traces] layout = "partitioned"`
   writes `traces/YYYY-MM-DD/<session_key>.jsonl`. Single-file modes remain
   (eval runner uses them). Startup stats rehydration globs partitions.
   Retention: `[log] retention_days` / `[traces] retention_days` pruned by
   the nightly job (not by the serving path).
5. **Derived index:** `scripts/analytics.py` (re)builds `harness.duckdb`
   with views over request partitions and trace partitions. DuckDB is an
   optional dependency (`[analytics]` extra); the serving path never
   imports it. Deleting the .duckdb file is always safe.

### Phase 1 — compose-native automation

- `flywheel` compose service: same image, command
  `python -m harness.flywheel`, shared `logs/`, `traces/`, `corpus/`,
  memory volumes. In-process asyncio scheduler (cheap, no cron dependency),
  jobs defined in `[flywheel]` config with cadences.
- Nightly: `memory_distill`, `corpus.py --include-live` append, partition
  retention, duckdb refresh, `compile_skills` refresh.
- Weekly: **envelope sentinel** — full n=20 run of all families against the
  live backend; per-family verdicts compared to the committed report; any
  downgrade from `supported` writes a loud run-report and flips a
  `sentinel_degraded` flag in `/stats`. The eval runner (and the `claude`
  CLI it drives) is bundled into the image for this.
- Every job appends one line to `logs/flywheel.jsonl` (job, started,
  duration, outcome, artifact paths) — the dashboard reads it.
- Memory persistence: named volume for the project-memory dir (today it
  dies with the container).

### Phase 2 — the learning leg

- `lora_train.py` grows backends: `--backend cuda` (QLoRA via
  peft/transformers on the RTX Pro 6000) and the existing `--backend mlx`
  (Apple silicon boxes). Same gated corpus in, adapter dir out.
- **Candidate serving via vLLM `--enable-lora`:** the candidate is the base
  27B plus an adapter, served by the existing vLLM instance under a new
  model name — no second GPU, no model copy. `harness.toml` gains the
  candidate as `roles = ["candidate"]` (router already isolates it).
- Flywheel threshold job: when the gated corpus grows past N new records,
  train → `shadow_eval` (full envelope against the candidate) →
  `promote_candidate` emits a proposed config diff. Promotion itself stays
  a human-reviewed commit until the loop has a track record (autonomy
  earned, umbrella principle 1).

### Phase 3 — the gate learns the real workload

- Trace-mined eval families: outcome-labeled live sessions become task
  templates (prompt, repo snapshot, mechanical checker derived from the
  verification command the session actually ran). Promotion certifies
  against synthetic + real families.
- `relax_scaffold` job: when a promoted model holds `supported` across two
  consecutive sentinels with a guard disabled in shadow, propose the
  relaxation diff. The system sheds its own training wheels.

## Acceptance criteria

- Phase 0: all tests green; live deploy on partitioned exhaust; dev-pr
  strings absent from `src/`; `analytics.py` answers "success rate by day"
  from partitions in one command.
- Phase 1: `docker compose up -d` yields serving + flywheel; a week of
  unattended operation produces nightly artifacts and one sentinel report
  with zero human actions.
- Phase 2: one adapter trained from live-gated corpus, shadow-evaled at
  n=20 with a verdict table, promotion diff generated automatically.
- Phase 3: ≥3 real-workload families in the gate; first eval-approved
  scaffold relaxation landed.

## Flagged assumptions (verify by measurement)

- QLoRA on the int4-servable base: adapters trained against the bf16 base
  apply cleanly to the int4-AutoRound deployment via vLLM LoRA (else the
  candidate serves from a bf16 copy on the 96 GB card — fits).
- The `claude` CLI runs headless inside the flywheel container against the
  local harness without cloud auth (it does in host evals today).
- 1–2 weeks of live traffic yields enough gated records (thousands) for a
  useful adapter; if not, Phase 2 waits on data, not on code.
- DuckDB-over-JSONL stays fast enough at 90-day retention scale (measured;
  else partition pruning tightens).
