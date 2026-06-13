# ai-harness: Small-LLM Platform Design

**Date:** 2026-06-12
**Status:** Approved direction; umbrella spec. Each capability below gets its own
spec → plan → implementation cycle.

## Vision

Replace cloud models for Claude Code workloads with local 4b–31b models, and keep
that true as the fleet grows and better models ship. The user experience must feel
like a cloud assistant — one endpoint, easy flow, no visible model juggling — while
every request is served by hyper-targeted use of the cheapest local capability that
can do the job.

Two reinforcing loops make this durable:

- **Runtime loop** (per request): shape prompts, surface tools, scaffold workflows,
  verify, and repair so a small model performs above its weight today.
- **Growth loop** (over weeks): traces → gated corpus → fine-tunes → eval gates →
  fleet promotion, so the models themselves improve and need less scaffolding
  tomorrow. New upstream model releases enter through the same gate as fine-tunes:
  everything is a candidate until evals promote it.

## Operating principles

1. **Codified first, autonomy earned.** Every decision starts as deterministic
   software. It is handed back to the model only when an eval proves that model
   holds it. The static-vs-LLM tradeoff is re-decided per capability, per model,
   by measurement — never by taste.
2. **Token thrift and prefix stability are architectural laws.** Anything injected
   into a prompt must be byte-stable across the turns of a session (it lives in the
   cached prefill) and must earn its tokens. Per-turn injection of varying content
   is forbidden on the hot path: it rewrites the prefix and forces a full re-prefill
   (measured cost on this fleet: ~20–60 s at 60k tokens).
   Corollary — **spend cheap resources to save scarce ones:** disk and CPU are
   comfortably managed; GPU time and model tokens are the finite resources. Offline
   jobs (skill compiling, memory distilling, corpus building, observability) may be
   disk/CPU-extravagant whenever that saves model tokens on the hot path.
3. **No capability ships without an eval delta.** `evals/` gates everything,
   including model adoption.
4. **Small surface is longevity.** One Python process, jsonl artifacts, TOML
   config. New infrastructure must displace more complexity than it adds.
5. **The system must be maintainable by its own models.** Frontier-model access is
   temporary; after it ends, a lower-tier LLM must be able to diagnose, patch, and
   extend this codebase. Every design choice is also judged by "could a 14b safely
   change this?" — see Self-maintainability below.

## Relationship to agentic-os

agentic-os (`/archive/agentic-os`) is an observability and recall substrate:
orchestrator + Postgres + Qdrant + LiteLLM + summarizer + embedder. It captures
trajectories, classifies requests, mediates tools, and packs recalled context. It
deliberately avoids learned routing and autonomous policy — it observes but never
closes the loop, and its per-request context enrichment is incompatible with
prefix-stability law (2).

The harness absorbs its proven ideas in harness idiom rather than running it:

| agentic-os concept | harness incarnation |
| --- | --- |
| Trajectories / event lineage | `traces/sessions.jsonl` + `logs/requests.jsonl` (exists) |
| Request classification | role detection, growing into capability routing |
| Tool mediation / menu shaping | tool autonomy (capability ①) |
| total-recall memory | project memory tier, cache-correct (capability ⑤) |
| Useful-deliverables-per-token | eval suite + per-capability counters (⑧) |

## Architecture

```
Claude Code / SDK / other clients
        │  (Anthropic API, one endpoint)
        ▼
┌─ ai-harness (single process) ─────────────────────────────┐
│ codec → pipeline stages → router → relay → profiles        │
│   ① tool autonomy   ④ planning scaffold   ⑥ guards         │
│   ② skills compiler ⑤ memory injection                     │
│   ③ research role   capability routing                     │
└────────────────────────────────────────────────────────────┘
        │ per-role / per-capability dispatch
        ▼
   fleet backends (vllm / llama.cpp), declared in harness.toml
        │ exhaust: traces, request log
        ▼
┌─ growth loop (offline jobs) ───────────────────────────────┐
│ corpus gate → fine-tune (LoRA) → eval gate → promotion     │
│ skill compiler · memory distiller · candidate shadow evals │
└────────────────────────────────────────────────────────────┘
```

## Capability map

### ① Tool autonomy (designed; first brick)

Selection (evolves `tool_prune.py`), priority order, cap `max_tools`:
1. every tool called anywhere in surviving history (first-call order — sticky and
   append-mostly for prefix stability);
2. tools named in the latest user turn (exact names, MCP server aliases, skill
   mentions → the `Skill` tool);
3. CORE fills remaining slots only (fixes the deadlock where CORE consumed the
   whole cap and MCP/Skill tools were never visible).

Catalog: a final system-prompt section lists the full tool inventory, one line per
tool (`name — purpose`), byte-identical across all turns of a session. The model
sees the whole toolbox; only ~8 get full schemas.

Schema swap (extends `relay.py`): a call to an unsurfaced tool is validated against
the full inventory — valid args pass through at zero cost; invalid args swap the
real schema in and retry under the existing `repair_retries` budget. Next turn the
tool is history-called and stays surfaced.

### ② Skills as compiled procedures

Skills are the codified-path library and the primary no-code extension mechanism.
An offline **skill compiler** rewrites each installed skill into a small-model
dialect: numbered imperative checklist, ≤400 tokens, no rationale prose; cached and
versioned per skill+model. At runtime a `Skill` invocation injects the compiled
form; checklist progress becomes scaffold state (④). Flagged assumption: compiled
skills retain enough fidelity to change small-model behavior — eval-verified early.

### ③ Research as a codified pipeline

Small models consume research; they do not conduct it. A `research` role runs
map-reduce retrieval: harness-driven fetch/search → chunk summarization fanned out
on the fast backend → one synthesized brief injected as a framed reference section.
Results cache into project memory so a question never costs twice.

### ④ Planning and execution scaffold

For multi-step tasks, a `plan` role (strongest local backend) produces a structured
step list (the SOW §6 "14b-executable" standard as runtime machinery). The harness
keeps the executor on rails with one compact live status line ("Step 3/7: …;
done: 1✓ 2✓") instead of trusting the model to remember the plan. Drift — edits
unrelated to the current step, done-claims with steps open — is detected
mechanically and fed back like a repair. Flagged assumption: qwen27-class models
produce usable structured plans; if not, planning is the one role with a temporary
cloud fallback while the growth loop catches up.

### ⑤ Memory, three tiers

- **Session:** deterministic compaction (exists), evolved append-mostly so
  compaction stops costing full re-prefills (monotonic eviction).
- **Project:** a per-repo fact store maintained by the harness itself. An offline
  post-session distiller extracts durable facts from traces ("tests run via
  .venv/bin/pytest", preferences, decisions) into a bounded, framed, byte-stable
  section injected at session start. Models read memory; the growth loop writes it,
  off the latency path. Stale-context framing (exists) applies.
- **Fleet:** KV residency and session affinity (exists).

### ⑥ Workflow guards

Extend the relay's guards (degenerate output, loop break — both exist) into a small
deterministic state machine over turn history: edit-without-read → nudge;
done-claim with no test/build since last edit → verify demand; same approach
failing twice → step-back prompt. Pure software, zero model dependence, highest
robustness-per-line in the design.

### ⑦ Growth loop automation & model adoption

Existing: trace capture, corpus gating, eval tasks. To build:
- **Fine-tune job:** LoRA per backend model consuming the gated corpus.
- **Candidate mechanism:** `roles = ["candidate"]` in `harness.toml`; candidates
  receive shadow eval runs (and optionally mirrored live traffic, never user-facing)
  against the incumbent on all task families.
- **Eval gate + promotion:** a candidate that beats the incumbent gets promoted by
  a config change; the diff is the audit trail. Fine-tunes and new upstream
  releases (fleet growth) use the identical path.
- **Scaffold relaxation:** when evals show a model no longer needs a guard or
  scaffold, it is relaxed for that backend (principle 1, in reverse).

### ⑧ Capability routing (multimodal, OCR, …)

Backends declare capability tags alongside roles: `vision`, `ocr`, `embed`, `audio`.
Requests are inspected for capability needs (image blocks → vision). Dispatch picks
the cheapest backend carrying the tag; when none exists, a **codified fallback**
runs — e.g. local OCR/captioning produces text injected as a framed block so
text-only models still serve the request. The user never sees the seams. This
generalizes role routing; roles become one kind of tag.

### ⑨ Observability

Per-capability counters in `logs/requests.jsonl`: `tool_surfaced`, `guard_fires`,
`plan_drift`, `memory_tokens`, `capability_fallbacks`. Every layer must prove its
delta continuously — the TTFT diagnosis of 2026-06-12 worked only because routing
already paid this debt.

## Self-maintainability (principle 5, concretely)

The harness is the system a small model will most often be asked to fix. Rules:

- **Plain Python, no cleverness.** No metaprogramming, no framework magic, no
  decorator stacks. Boring code a 14b can hold in one read. (This is also why
  agentic-os is mined rather than run: a 14b cannot safely patch a Rust
  orchestrator plus SQL migrations plus a vector store.)
- **Small files, one purpose.** Pipeline stages stay ~100 lines with the invariant
  stated in the module docstring. A file that outgrows one read gets split.
- **Tests are the safety harness for the maintainer-model.** Every behavior has a
  test (147 today); a small model verifies its patch by running them, not by
  reasoning about correctness. TDD remains mandatory precisely because the future
  maintainer cannot be trusted to reason — only to run.
- **Diagnostic runbooks as skills.** Recurring investigations are codified as
  step-by-step runbooks (e.g. "TTFT regression: compute percentile chunks from
  `logs/requests.jsonl`, check cached_tokens collapse, check session bouncing,
  check compaction churn" — the 2026-06-12 diagnosis, written down). A 14b with a
  runbook reproduces today's frontier-model debugging; without one it guesses.
  Runbooks live with skills and go through the skill compiler (②).
- **Specs and plans stay in-repo** (`docs/superpowers/specs/`, `docs/plans/`),
  written 14b-executable, so intent survives next to code.
- **Self-diagnosis before self-modification.** The observability counters (⑨) are
  designed to localize faults to a capability, so the maintainer-model starts from
  evidence, never from a cold codebase search.

## UX invariants (the "cloud experience", testable)

1. **One endpoint.** Clients see a single Anthropic-compatible API; fleet
   composition is invisible.
2. **Never a dead end.** Degradation chain: warm backend → role overflow →
   capability fallback → honest structured error. No silent drops.
3. **Latency budgets per role:** fast ≤2 s TTFT p50; main ≤5 s warm; budget
   violations are logged and eval-tracked.
4. **No user-visible model juggling.** Capability seams (OCR fallback, research
   fan-out, plan/execute split) never require user awareness.

## Extensibility points

Extending the system means adding one of these; the core is never forked:

| Point | How |
| --- | --- |
| Model | `[[backends]]` entry as candidate → eval gate → promote |
| Dialect | profile (chat template / tool-call parsing) |
| Modality | capability tag + codified fallback |
| Tool | MCP server / client tool; auto-enters catalog ① |
| Procedure | skill; auto-compiled by ② |
| Codified path | pipeline stage or relay guard, with eval family |
| Quality gate | eval task family in `evals/tasks/` |
| Growth job | offline script over traces/corpus |

## Build order

1. **Tool autonomy** ① — designed; unblocks skills and MCP.
2. **Eval expansion** — `tool-discovery` + `long-horizon` task families; ④⑤⑥ all
   claim to fix long-task drift and need a ruler first.
3. **Workflow guards** ⑥ — cheapest, highest yield.
4. **Planning scaffold** ④.
5. **Project memory** ⑤.
6. **Skill compiler** ②.
7. **Capability routing + first fallback (OCR)** ⑧.
8. **Research pipeline** ③.
9. **Growth-loop automation + candidate adoption** ⑦ — data collection already
   runs; automate fine-tune/eval-gate/promotion last, then it runs forever.

Every implementation plan is written 14b-executable (SOW §6).

## Flagged assumptions (verify early, by eval)

- Compiled ≤400-token skills change small-model behavior enough to matter (②).
- qwen27-class models produce usable structured plans (④).
- Catalog awareness (~350 stable tokens) measurably improves tool selection rather
  than distracting 4b–9b models (①) — the eval may set different catalog sizes per
  model class.
- A local OCR/captioning stage is good enough to serve vision requests through
  text-only models for the common cases (⑧).
