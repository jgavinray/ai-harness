# Platform Build Roadmap

Master task index for the umbrella spec
(`docs/superpowers/specs/2026-06-12-small-llm-platform-design.md`).
Each row becomes its own detailed, 14b-executable plan in this directory when its
turn comes — written while frontier-model access lasts, executed by whatever model
is available. **Rule: a plan is written against the codebase as it exists then;
never execute a stale plan without re-reading the files it names.**

| # | Capability | Plan file | Status |
|---|-----------|-----------|--------|
| 1 | Tool autonomy (①) | `2026-06-12-tool-autonomy.md` | **done 2026-06-12** (b47a2fb..9c35752, 162 tests green) |
| 2 | Eval expansion | `2026-06-12-eval-expansion.md` | **done 2026-06-12** |
| 3 | Workflow guards (⑥) | `2026-06-12-workflow-guards.md` | **done 2026-06-12** |
| 4 | Planning scaffold (④) | `2026-06-12-planning-scaffold.md` | **done 2026-06-13** |
| 5 | Project memory (⑤) | `2026-06-12-project-memory.md` | **done 2026-06-13** |
| 6 | Skill compiler (②) | `2026-06-12-skill-compiler.md` | **done 2026-06-13** |
| 7 | Capability routing + OCR fallback (⑧) | `2026-06-12-capability-routing.md` | **done 2026-06-13** |
| 8 | Research pipeline (③) | `2026-06-12-research-pipeline.md` | **done 2026-06-13** |
| 9 | Growth loop + candidate adoption (⑦) | `(to write)` | — |

## Plan-writing checklists (what each future plan must cover)

**2 — Eval expansion.** Read `evals/run.py` and one task dir (`evals/tasks/fix-test/`:
`prompt.txt` + `check.sh` + `repo_template/`) first. Add `tool-discovery` family
(task solvable only via a non-CORE tool; harness must surface it) and
`long-horizon` family (10+ step task; measures drift — the ruler for plans 3-5).
Add per-model catalog-size ablation (does the catalog distract a 4b?). Wire
`tool_surfaced`, `guard_fires` counters into `evals/report.py`.

**3 — Workflow guards.** Deterministic checks over turn history in the relay:
edit-without-read nudge; done-claim with no test/build since last edit → verify
demand; same approach failed twice → step-back feedback. Each guard: one function,
one counter (`guard_fires.<name>`), one test file, individually toggleable in
`[pipeline]`. Pattern to follow: `LOOP_THRESHOLD` cross-turn loop-break in
`src/harness/relay.py` (d010594).

**4 — Planning scaffold.** `plan` role in router; structured step list produced
once per task; one compact status line injected at a STABLE prompt position (end
of system prompt; never per-turn varying mid-prompt — prefix law). Mechanical
drift detection feeding the repair channel. Verify qwen27 plan quality by eval
before building the injection machinery (flagged assumption in spec).

**5 — Project memory.** Offline distiller (`scripts/` job over `traces/`) writing
per-repo facts; injection reuses `MemoryStage` (`src/harness/memory.py`), bounded
by `memory.max_chars`, framed by the existing stale-context framing in
`system_prompt.py`. Byte-stable per session: distill between sessions, never
during one.

**6 — Skill compiler.** Offline job: installed skill → ≤400-token imperative
checklist, cached per (skill, model-class), versioned by content hash. Runtime:
`Skill` tool call returns compiled form. Eval: does a compiled skill change
task outcomes on the long-horizon family?

**7 — Capability routing + OCR.** `capabilities: list[str]` on `PoolBackendCfg`
(pattern: `roles`); request inspection for image blocks in the codec; codified
fallback chain (OCR → text injection as framed block). Pick the OCR engine by
benchmarking locally first (CPU is cheap; model tokens are not).

**8 — Research pipeline.** `research` role; map-reduce: fetch → chunk →
fast-backend summaries → one synthesized brief; cache briefs into project memory
keyed by query hash. Never inject raw retrieval into the main conversation
(prefix law + token thrift).

**9 — Growth loop + adoption.** `roles = ["candidate"]` backends excluded from
live routing; shadow eval job; LoRA fine-tune job over `evals/results/corpus.jsonl`;
promotion = eval-gated `harness.toml` edit. Also: scaffold relaxation — per-backend
guard/scaffold toggles informed by eval results.

## Standing constraints for every plan

- TDD; suite green after every task; commits per task.
- Prefix stability: nothing injected may vary per-turn within a session.
- Token thrift: spend disk/CPU to save model tokens, never the reverse.
- Self-maintainability: plain Python, files ~≤120 lines or split, invariants in
  docstrings, runbooks for recurring diagnoses.
- Plans must be 14b-executable: exact paths, complete code in steps, exact
  commands with expected output (SOW §6).
