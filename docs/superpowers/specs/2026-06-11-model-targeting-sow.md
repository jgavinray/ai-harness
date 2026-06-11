# Scope of Work: Model Targeting & Downscale Path

**Date:** 2026-06-11
**Companion to:** `2026-06-11-small-llm-claude-code-proxy-design.md`
**Purpose:** Define which models the harness targets, how we measure effectiveness per model, and the concrete path for using the harness itself to bootstrap a similar system on a much lower-tier model.

## 1. Model tiers

### Tier 1 — Primary targets (14b–30b): the harness's design band

| Model | Size | Tool calling | Context | Notes |
|---|---|---|---|---|
| Qwen2.5-Coder-32B-Instruct | 32B dense | Strong, native | 32k (128k YaRN) | Reference model; tune the rewrite prompt here first |
| Qwen3-Coder-30B-A3B | 30B MoE (3B active) | Strong | 256k | Fastest per-token in tier; great daily driver |
| Devstral Small | 24B dense | Strong, agent-trained | 128k | Purpose-built for coding-agent harnesses; expect best edit discipline |
| DeepSeek-R1-Distill-Qwen-14B/32B | 14/32B | Weak–moderate | 32k | Reasoning via `<think>`; planning strength, tool-format weakness — the repair loop earns its keep here |
| Qwen2.5-Coder-14B | 14B dense | Good | 32k | Lower-bound of design band |
| Gemma 3 27B | 27B dense | Moderate (no native tool role) | 128k | Exercises the no-system-role / prompted-tools path |

### Tier 2 — Downscale targets (4b–9b): the "mimic" experiment band

Qwen2.5-Coder-7B, Qwen3-8B / Qwen3-4B, Gemma 3 12B/4B, Llama-3.1-8B-Instruct.
Expectation: these fail as drop-in replacements without harness changes; they are the subjects of Phase D below.

### Tier 3 — Floor probes (1b–3b)

Qwen2.5-Coder-3B/1.5B, Gemma 3 1B. Not expected to complete sessions; used to find which pipeline features degrade gracefully and to size the limits of the approach.

## 2. Per-tier harness posture

The same pipeline, progressively stricter as models shrink:

| Lever | Tier 1 (14–30b) | Tier 2 (4–9b) | Tier 3 (1–3b) |
|---|---|---|---|
| System prompt | replace (~1.5k tok) | replace, shorter (~800 tok) | minimal (~300 tok) |
| Max tools | 8 | 5–6 | 3 (Read/Edit/Bash) |
| Few-shot examples | 2–3 | 3–4 (more critical) | inline per-turn |
| Constrained decoding | on repair retry | **always on** for tool args | always on |
| Repair retries | 2 | 3 | 3 |
| Plan injection (future stage) | off | on for multi-step tasks | mandatory |
| Tool-result truncation | 1500 chars | 800 chars | 400 chars |

These become named **presets** in `harness.toml` (`tier1` / `tier2` / `tier3`) once Phase C data says where the real cliffs are.

## 3. Evaluation plan

### Task suite (scripted, reproducible Claude Code sessions)

1. **fix-test** — repo with one failing pytest; success = test passes.
2. **add-endpoint** — add a route + handler to a small FastAPI app; success = new test passes.
3. **rename-refactor** — rename a function across 3 files; success = grep finds zero old references, tests pass.
4. **find-and-report** — locate where a config value is read (read-only task); success = correct file:line in final answer.
5. **multi-step** — fix-test that first requires installing a missing dep (forces Bash + diagnosis chaining).

Each runs as a real `claude -p "<task>"` invocation through the proxy against a throwaway git repo, 3 trials per model.

### Metrics (from harness structured logs — no extra instrumentation needed)

- Task success rate (did the end state pass the checker)
- Invalid-tool-call rate, pre- and post-repair (the harness's core value metric)
- Repair retries consumed per session; sessions wedged/degraded to text
- Tokens in/out per task; wall-clock per task; time-to-first-token
- Context-budget evictions triggered

### Ablations

Each pipeline stage toggled off individually on the reference model (Qwen2.5-Coder-32B) and the weakest Tier 1 model (R1-Distill-14B) to quantify per-stage contribution. This tells us which stages matter as size drops.

## 4. Downscale path: using the harness to build the mimic system

The harness is the data engine, not just the runtime. `dump_prompts` already captures, per request: the exact rendered backend prompt, the model's full response, repairs triggered, and the outcome.

**Phase D1 — Trace capture.** Run the eval suite + real working sessions on Tier 1 models with `dump_prompts=true`. Result: a corpus of (rendered prompt → correct agentic response) pairs, in the small model's own chat format, including tool calls that *validated* (post-repair filtering = free quality labeling).

**Phase D2 — Behavioral cloning eval.** Replay captured prompts against Tier 2 models offline (no Claude Code in the loop) and diff: did the small model pick the same tool? Valid args? This isolates model capability from harness behavior and is cheap to run across many candidates.

**Phase D3 — Tier 2 presets.** Apply the stricter posture from §2, rerun the live eval suite, find the best Tier 2 model + preset combination. Acceptance: ≥70% of the reference model's task success on the suite.

**Phase D4 (stretch) — Fine-tune the mimic.** LoRA SFT of the best Tier 2 model on the filtered Phase D1 corpus (successful sessions only). Because the corpus is rendered through the harness, the fine-tuned model trains on the *exact* prompt distribution it will see in production — this is the strongest version of "use the harness to build a similar system on a lower-tier model." Re-run Phase D3 acceptance afterward.

## 5. Deliverables & sequencing

| Phase | Deliverable | Depends on |
|---|---|---|
| A | Harness v1 (current implementation plan) | — |
| B | Eval suite repo (`evals/` dir: 5 task repos + runner script + metrics report) | A |
| C | Tier 1 model report (matrix of metrics + ablations) + tuned per-family rewrite prompts | B |
| D1 | Trace corpus + capture tooling (filter/format scripts) | C |
| D2–D3 | Tier 2 preset configs + downscale report | D1 |
| D4 | Fine-tuned mimic model + final comparison | D3 (optional) |

Phase A is in progress now. Phases B+ each get their own brainstorm → spec → plan cycle when reached; this SOW is the standing map, not a commitment to D4.
