# Daily-Driver Consistency: Closed-Loop Turn Contract

**Date:** 2026-07-07
**Status:** Proposed. Sub-project of the umbrella spec
(`2026-06-12-small-llm-platform-design.md`); supersedes nothing, sequences the
umbrella's remaining bricks against a concrete product bar.

## Goal

qwen3.6-27b (or 35b, decided by bake-off) serves as an unattended Claude Code
daily driver. Value is defined as: consistency and high-level execution of
tasks without the operator babysitting the harness.

The bar is **Opus-like reliability inside a declared envelope**, not Opus-parity
single-shot intelligence:

- no session ever dies silently or ends in garbage;
- "done" always means the relevant check ran and passed;
- tasks beyond the model end in an honest structured failure with evidence,
  never silent wrongness;
- wall-time stays at or below today's `full` config.

## Reality (evidence, reviewed 2026-07-07)

Per-task decomposition of `evals/results/**` (never done before — `report.py`
aggregates only by model+config) shows the 0.80 success rate is **not** a
model-intelligence number. Every failure on disk has one of two signatures:

1. **Give-up death** (fix-test, multi-step; flaky across audit reruns).
   First tool call fails validation → one repair retry → model emits a sentence
   of prose and stops with `end_turn` and zero tool calls → `claude -p` accepts
   that as a final answer → 4-second dead session. The trace also records
   0 input / 0 output tokens despite streamed text (usage accounting bug on the
   retry path). The uncommitted working-tree diff (`_vllm_grammar_schema`,
   `propertyNames` in NOISE_KEYS, critic-reviewed preflight) chases this chain.
2. **Silent wrong completion** (rename-refactor; reproducible in baseline AND
   full). 10–13 requests of real work, ends with `compute_total not defined` —
   the model never ran the check before finishing. Verify-after-edit exists as
   a nudge (`guards.py`), not a gate.

When neither signature fires, qwen27 passes 7/7 task families in both configs.
n=1 per task per config means one flake moves aggregate success 14 points: the
current ruler cannot distinguish a fix from noise.

Separately: the live `harness.toml` runs `system_prompt = "passthrough"`,
`tool_prune = false`, critic disabled, single backend — most of the built
machinery is dark in production.

**Conclusion:** the model is mostly adequate; the harness lets bad turns
escape. Both failure classes are consistency failures, which is the product
bar. This spec fixes the loop, not the model.

## Core principle: the harness is a closed-loop controller

Today the pipeline is an input shaper: it perfects the prompt, then relays
whatever comes back. The inversion that removes babysitting:

> **Turn-completion contract.** No model turn reaches the client unless it
> satisfies the current action state's exit criteria. A violating turn is fed
> back as structured feedback (like repair), bounded by budget; budget
> exhaustion triggers the escalation ladder, whose terminal state is an honest
> structured failure — never a silent relay.

The model is a proposal generator inside a loop that cannot emit garbage.
Prompt shaping, constrained decoding, planning, routing, and fine-tuning all
feed this loop; none of them replaces it.

### The contract, concretely

| State (from `action_state.py`) | Exit criteria for the turn | On violation |
| --- | --- | --- |
| tool-required (task incomplete, work expected) | ≥1 valid tool call | feed back "you ended without acting; call a tool or declare failure" |
| unverified-edit | verification tool call (test/build/check) before any done-claim or `end_turn` | feed back verify demand; the existing guard wording, now blocking |
| done-claim | the task's check command ran since the last edit and passed | feed back the failing check output |
| stuck (same approach failed twice — `guards.py` repeat detection) | a different approach, or escalation | enter escalation ladder |
| any | no degenerate/looping output (exists) | existing loop-break, unchanged |

Budgets: per-turn feedback retries reuse `repair_retries` semantics; per-session
contract-feedback total is capped (config, default 8) so a hopeless task
converges to honest failure instead of burning tokens.

### Escalation ladder (bounded test-time compute)

When a step exhausts its budget, escalate in order; each rung is optional per
config and skipped when its backend is absent:

1. **Resample:** N=3 candidates at temperature on the main backend; critic role
   ranks; best proceeds. (Uses existing `critic.py` machinery.)
2. **Stronger local model:** re-run the step on a `reasoning`/`plan` role
   backend (qwen80-thinking when up).
3. **Honest structured failure:** an Anthropic-shaped final message stating
   what was attempted, what failed, and the evidence (failing check output).
   This is a *success* for the product bar: visible gap, no silent wrongness.

## Components (mapped to existing code)

| # | Component | Status | Where |
| --- | --- | --- | --- |
| 1 | Turn contract enforcement | **harden** | `relay.py` + `action_state.py`: recent commits enforce tool-required states preflight; extend to cover the post-repair-retry path and `end_turn`-without-action; done-claim gate uses `guards.py` predicates but blocks instead of nudging |
| 2 | Usage-accounting fix | **bug** | retry path loses the vLLM usage chunk → Done carries 0 tokens; fix in `profiles/base.py` parse or `relay.py` accumulation |
| 3 | First-attempt decode constraints | **new (small)** | `VllmBackend.apply_constraint` exists for retries; apply grammar constraint on the initial call for tool-capable requests so failure class 1 loses its trigger |
| 4 | Escalation ladder | **new (small)** | thin orchestration over existing critic + role routing + reasoning budgets |
| 5 | Ruler v2 | **extend** | `evals/report.py`: per-task rows; mechanical failure tagging (zero-token → `infra-death`, work-done-check-failed → `wrong-result`, `timeout`); n≥20 per family; contract counters (`contract_feedback`, `gave_up_honestly`, `died_silently`) added to `log.py` records and aggregated |
| 6 | 27b vs 35b bake-off | **run, not build** | same suite, both backends; decides `main`. Memory note: the 35B heretic MoE had the cleanest tool calls — quantify it |
| 7 | Relight dark machinery | **config + measurement** | one feature at a time (`system_prompt=replace`, `tool_prune`, critic, planning), each kept only on a ruler delta; resolve `policy_owner`: eval/live Claude Code traffic is harness-owned |
| 8 | Growth loop (umbrella ⑦) | **after ruler** | LoRA distillation of gated frontier traces; its eval gate is only as trustworthy as the ruler |
| 9 | Throughput pass | **last** | component 3 already deletes retry round-trips; then FP8 / speculative decoding on the RTX Pro 6000; MoE placement falls out of the bake-off |

## Acceptance criteria (product-level, measured at n≥20 per family)

- `died_silently` = 0 across all runs (give-up deaths eliminated).
- `wrong-result` completions = 0 for supported families; failures become
  `gave_up_honestly` with evidence.
- Success ≥ 0.95 per family declared **supported**; the per-family verdict
  table (supported / degraded / unsupported) ships with the report — this is
  the declared envelope.
- Wall-time per session ≤ today's `full` config.
- Zero human interventions required during any eval session (by construction —
  the loop cannot stall waiting for a human).

## Build order

1. Contract enforcement + usage fix (components 1, 2) — closes both observed
   failure classes; finishes the in-flight working-tree work.
2. Ruler v2 (component 5) — small code; every later brick is gated by it.
3. First-attempt constraints (3).
4. Bake-off (6), then relight machinery feature-by-feature (7).
5. Escalation ladder (4).
6. Growth loop (8), throughput pass (9).

Each brick gets its own 14b-executable plan (SOW §6).

## Flagged assumptions (verify by eval, not taste)

- vLLM's structured-output / structural-tag support on the deployed version
  works with Qwen 3.6's chat template for *first-attempt* tool constraints; if
  not, fallback is forced named `tool_choice` on tool-required states.
- Contract feedback loops converge for qwen27-class models rather than
  oscillating; the per-session cap bounds the cost of being wrong.
- Critic ranking of resampled candidates beats first-sample quality enough to
  justify rung 1 of the ladder (measured, else the rung is removed).
- The 7 task families represent daily-driver work well enough for the envelope
  claim; add families where real usage diverges (traces show what real
  sessions do).

## Non-goals

- Raw Opus-parity single-shot capability on a 27b.
- Cloud fallbacks on the serving path (umbrella permits a temporary plan-role
  exception; this spec does not use it).
- New infrastructure: no new processes, stores, or frameworks. Every component
  is a bounded change to an existing module (umbrella principle 4).
