# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An Anthropic Messages API proxy that lets Claude Code run against small local
models (vLLM / llama.cpp): it decodes Anthropic requests into a neutral IR,
runs pipeline stages that make the request survivable for a 14–30B model,
relays to an OpenAI-compatible backend with validation/repair/contract
enforcement, and encodes Anthropic responses back. A second process (the
flywheel) runs the self-improvement loops. Read
`docs/source-walkthrough.md` before making non-trivial changes — it maps every
module and lists where to start for common change types.

## Commands

```bash
.venv/bin/pytest -q                       # full suite; must be green before any commit
.venv/bin/pytest tests/test_relay.py -q   # one file
.venv/bin/pytest tests/test_relay.py::test_happy_path -q   # one test
docker compose build && docker compose up -d   # the ONLY deployment path (serving + flywheel)
.venv/bin/python evals/run.py --backend-url http://192.168.0.196:8000/v1 \
    --model qwen3.6-27b --profile qwen --kind vllm \
    --configs full --trials 20 --out evals/results/<name>   # envelope eval (add --tasks a,b to filter)
.venv/bin/python evals/report.py evals/results/<name>/results.jsonl   # per-family verdict table
.venv/bin/python scripts/analytics.py query "SELECT ... FROM requests"  # SQL over the data plane
docker compose exec flywheel python -m harness.flywheel --config /config/harness.toml --once nightly
```

## Non-negotiable laws (from docs/superpowers/specs/2026-06-12-small-llm-platform-design.md)

1. **No capability ships without an eval delta.** The envelope suite
   (`evals/`, 7 task families, n=20, verdicts supported/degraded/unsupported
   at 0.95/0.80) gates every feature, model, and config change. History:
   every eval failure so far was a harness gate fighting legitimate model
   behavior, not model weakness — measure before assuming either.
2. **Prefix stability.** Anything injected into a prompt must be byte-stable
   across a session's turns; per-turn varying injection forces a full
   re-prefill (20–60 s at 60k tokens). Pipeline stages must be deterministic.
3. **TDD is mandatory** — the future maintainer may be a small model that can
   only run tests, not reason about correctness. Write the failing test
   first; it should reproduce the observed failure (eval trace, log record).
4. **14b-maintainable code**: plain Python, no metaprogramming, small files
   with the invariant in the module docstring, wire formats only in
   codecs/profiles, transport only in backend classes, request shaping only
   in pipeline stages.
5. **JSONL is the source of truth.** `logs/requests/YYYY-MM-DD.jsonl` and
   `traces/YYYY-MM-DD/<session>.jsonl` are the data plane; the DuckDB file is
   a derived, disposable index. Never add an always-on service beyond the two
   compose containers.
6. Word-trigger heuristics over free-form prompt text are guilty until
   eval-proven (recurring bug shape: long task briefs locking the tool
   surface). Free-text intent may only bind for short imperative
   instructions (`SHORT_INSTRUCTION_MAX_CHARS`).

## Architecture (the parts that span multiple files)

- **Request path** (`server.py`): Anthropic wire → `codec/anthropic_in` → IR
  (`ir.py`) → `Router.pick()` (before the pipeline: compaction needs the
  backend's context window) → pipeline stages (`pipeline/`: system prompt
  rewrite, path canon, tool prune/catalog, schema trim, history compaction,
  fewshot, memory) → profile render (`profiles/`) → `relay.run()` →
  `codec/anthropic_out`. Wire formats never leak past the codecs/profiles.
- **The relay is a closed-loop controller, not a pass-through**
  (`relay.py` + `action_state.py` + `guards.py` + `repair/`): tool calls are
  validated against the ORIGINAL schema and repaired (string-scalar coercion
  included); deterministic guards and action states gate what a turn may do
  (verify state binds after `unverified_edit_limit` edits, done-claims
  require verification); a turn that gives up after tool-retry feedback is
  fed back, and exhausted budgets end in an explicit `[harness]` failure —
  never a silent empty turn. Streams that stall past
  `stream_idle_timeout_s` end honestly too. Usage accumulates across retry
  attempts. See `docs/superpowers/specs/2026-07-07-daily-driver-consistency-design.md`.
- **Flywheel** (`flywheel.py`, second compose service, same image): nightly
  memory distill / gated-corpus rebuild / partition retention / DuckDB
  refresh / skill compile; weekly envelope sentinel that re-runs the eval
  suite and writes `logs/flywheel_sentinel.json` (surfaced in `/stats`).
  Jobs are subprocesses of `scripts/`; a job failure never touches serving.
  Roadmap: `docs/superpowers/specs/2026-07-09-self-improving-flywheel-design.md`.
- **Config** (`config.py` + `harness.toml`): the live `harness.toml` mirrors
  the eval-certified `full` config — its header explains the provenance.
  Code defaults ARE certified values; only add config lines the evals have
  measured. `candidate` role isolates backends from live traffic.
- **Evals** (`evals/run.py`) drive the real `claude` CLI against a fresh
  harness per trial; per-trial outcome + request-log metrics join into
  `results.jsonl`; `report.py` classifies failures mechanically
  (infra-death / timeout / honest-give-up / wrong-result).

## Where truth lives

- `docs/superpowers/specs/` — approved designs (umbrella → consistency →
  flywheel). `docs/superpowers/plans/` — 14b-executable implementation plans.
- `docs/reports/` — measured evidence; claims about quality come from here,
  not from memory.
- `docs/superpowers/runbooks/improvement-loop.md` — step-by-step improvement
  loop for a small-model maintainer.
- Commit messages carry the evidence for each fix (eval run + trace
  signature); `git log` is the debugging history.
