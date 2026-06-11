# ai-harness: Claude Code–Compatible Proxy for Small LLMs — Design

**Date:** 2026-06-11
**Status:** Approved
**Goal:** Let the real Claude Code CLI run seamlessly against a local 14b–30b model by placing an optimizing proxy between them.

## 1. Overview

`ai-harness` is a single Python process: a FastAPI server that implements the
subset of the Anthropic Messages API that Claude Code uses, and forwards work
to any OpenAI-compatible inference backend. Usage:

```bash
ANTHROPIC_BASE_URL=http://localhost:8484 ANTHROPIC_API_KEY=local claude
```

The proxy treats Claude Code's requests as *intent, not gospel*: it rewrites
the system prompt, prunes and simplifies tools, compacts history, and injects
few-shot examples before the request reaches the small model — then validates
and repairs the model's output so that Claude Code always receives a
spec-valid Anthropic response.

### Decisions already made

- **Shape:** API proxy for the real Claude Code CLI (not a standalone harness).
- **Language:** Python 3.12+, FastAPI, httpx, async SSE end to end.
- **Downstream protocol:** OpenAI Chat Completions as the universal baseline,
  with capability extensions for vLLM (guided/structured decoding) and
  llama.cpp server (GBNF grammars).
- **Transform policy:** full rewrite pipeline (not conservative passthrough).
- **Target model families (in priority order):** Qwen coder family
  (Qwen2.5-Coder-14B/32B, Qwen3-Coder-30B-A3B), DeepSeek-R1 distills
  (14B/32B), Devstral Small 24B, Gemma family.
- **v1 scope:** core agentic loop — streaming SSE, full tool-use round trips,
  multi-turn context management, `count_tokens`, sub-agent (Task tool)
  concurrency. Deferred: image blocks, web search/fetch passthrough,
  citations, fine-grained prompt-cache emulation.

## 2. API surface (v1)

| Endpoint | Behavior |
|---|---|
| `POST /v1/messages` | Streaming (SSE) and non-streaming. The main path. |
| `POST /v1/messages/count_tokens` | Answered locally via the active model profile's HuggingFace tokenizer; no backend call. |
| `GET /stats` | Session counters (requests, repairs, token usage). Not part of the Anthropic API; for the operator. |

- All error responses follow the Anthropic error envelope
  (`overloaded_error`, `api_error`, `invalid_request_error`) so Claude Code's
  built-in retry/backoff works unchanged.
- Auth: any `x-api-key` is accepted (local, single-user).
- Sub-agents need no special handling: Claude Code issues them as additional
  concurrent `POST /v1/messages` requests. The server is fully async and
  stateless per request.

## 3. Core abstraction: the IR

Every request is decoded into an **intermediate representation** — typed,
frozen dataclasses modeling the conversation neutrally: system text, turns,
tool definitions, tool calls, tool results, generation parameters. The IR is
neither Anthropic-shaped nor OpenAI-shaped.

- **Codecs** (Anthropic decode/encode) touch wire format exactly once, at the
  edges.
- **Pipeline stages** transform IR → IR and never see wire formats.
- **Model profiles** render IR → backend request, and parse backend stream
  events → IR events.

This isolates four model families × ~8 optimizations × 3 backends into
independently testable units.

## 4. Components

```
src/harness/
  server.py              # FastAPI app, SSE plumbing, Anthropic error mapping
  config.py              # pydantic-settings: backend, profile, stage toggles, budgets
  ir.py                  # intermediate representation (frozen dataclasses)
  codec/
    anthropic_in.py      # Anthropic request JSON → IR
    anthropic_out.py     # IR stream events → Anthropic SSE / JSON response
  pipeline/              # ordered IR → IR stages, each toggleable via config
    system_prompt.py     # ① system prompt rewrite
    tool_prune.py        # ② per-turn tool subsetting
    tool_schema.py       # ③ schema simplification
    history.py           # ④ token-budget compaction
    fewshot.py           # ⑤ tool-call example injection
  profiles/              # ModelProfile protocol + implementations
    base.py              #   tokenizer, stop seqs, tool-call wire format,
    qwen.py              #   reasoning-token handling, chat-template quirks
    deepseek_r1.py
    devstral.py
    gemma.py
  backends/
    base.py              # Backend protocol (async stream of completion deltas)
    openai_compat.py     # universal baseline (Ollama, LM Studio, OpenRouter…)
    vllm.py              # + guided_json structured outputs
    llamacpp.py          # + GBNF grammar constraints
  repair/
    toolcalls.py         # ⑥ JSON repair → schema validation → bounded retry
    degenerate.py        # ⑦ repetition / runaway-loop detection
  tokens/
    counter.py           # tokenizer counting for count_tokens + history budget
```

## 5. Request flow

```
Claude Code request
  → codec.anthropic_in (request JSON → IR)
  → pipeline stages ①–⑤ (IR → IR)
  → profile.render (IR → backend request, incl. constrained-decoding params)
  → backend.stream (async deltas)
  → profile.parse (deltas → IR events)
  → repair ⑥⑦ (validate; may re-call backend with error feedback, max 2)
  → codec.anthropic_out (IR events → Anthropic SSE)
  → Claude Code
```

## 6. Optimization pipeline

### Request side

1. **System prompt rewrite** (`replace` | `compress` | `passthrough`;
   default `replace`). Claude Code's 10–20k-token system prompt is detected
   and substituted with a ~1.5k-token prompt tuned for small models:
   imperative style, the behavioral contracts that matter (tool-use rules,
   edit-format discipline, conciseness, safety), nothing else. User CLAUDE.md
   content embedded in the incoming prompt is preserved verbatim. Detection is
   by structural fingerprint (known section headings), not exact match, so it
   survives Claude Code version bumps; unrecognized system prompts fall back
   to `compress`.
2. **Tool pruning** (`max_tools`, default 8). Core set always kept: Read,
   Edit, Write, Bash, Grep, Glob, TodoWrite, Task. Other tools included only
   if referenced within the protected recent-turn window (same
   `recent_turns_protected` setting as history compaction). Pruned tools are
   simply not rendered; the
   model cannot call what it cannot see, and Claude Code is indifferent.
3. **Schema simplification.** Flatten nested JSON schemas, shorten verbose
   descriptions, drop rarely-used optional parameters. The repair stage
   re-validates against the **original** schema and fills safe defaults, so
   tool_use blocks returned to Claude Code always satisfy the original
   contract.
4. **History compaction.** Token-budget manager using the profile's real
   tokenizer. When over budget, in order: truncate old tool results to
   head+tail with an explicit elision marker; then evict oldest complete
   turns. The system prompt and the most recent K turns (default 4) are never
   modified. Fully deterministic — no LLM summarization calls in v1.
5. **Few-shot injection.** 2–3 canonical tool-call examples in the profile's
   exact wire format, appended to the rewritten system prompt. Cheapest
   single reliability win for small models.

### Response side

6. **Constrained decoding** (capability-gated; configured at render time but
   listed here because it governs output validity). On vLLM, `guided_json`; on
   llama.cpp, a GBNF grammar compiled from the simplified tool schemas. On
   plain OpenAI-compatible backends this is a no-op and stage 7 carries the
   load.
7. **Repair loop.** Parse tool calls → `json_repair` for almost-valid JSON →
   validate against the original Anthropic schema → on failure, re-call the
   backend with the validation error appended as feedback (max 2 retries,
   configurable) → if still failing, emit the model output as plain text so
   the session never wedges. Degenerate-output detection (n-gram repetition
   over a sliding window) aborts a runaway stream and triggers one retry at
   higher temperature.
8. **Reasoning-token handling.** R1-class `<think>…</think>` spans are mapped
   to Anthropic `thinking` content blocks (Claude Code renders them natively)
   or stripped, per config (`reasoning = "thinking" | "strip"`).

### Streaming policy

Text deltas stream through with minimal added latency. Tool-call argument
deltas are **buffered until validated/repaired**, then emitted as a burst of
`input_json_delta` events. A tool call must be provably well-formed before
Claude Code sees any of it — the deliberate latency-for-correctness trade.

## 7. Configuration

Single TOML file (`harness.toml`) + env-var overrides via pydantic-settings:

```toml
[server]
host = "127.0.0.1"
port = 8484

[backend]
kind = "openai"            # openai | vllm | llamacpp
base_url = "http://localhost:11434/v1"
model = "qwen2.5-coder:32b"

[profile]
name = "qwen"              # qwen | deepseek_r1 | devstral | gemma
context_window = 32768

[pipeline]
system_prompt = "replace"  # replace | compress | passthrough
tool_prune = true
max_tools = 8
fewshot = true
repair_retries = 2
recent_turns_protected = 4
reasoning = "thinking"     # thinking | strip

[debug]
dump_prompts = false       # write rendered backend prompts to disk
```

## 8. Error handling

| Failure | Behavior |
|---|---|
| Backend unreachable / 5xx | Anthropic `overloaded_error` (529) → Claude Code retries with backoff |
| Mid-stream backend death | Emit valid Anthropic `error` SSE event, close the event sequence cleanly |
| Repair retries exhausted | Degrade tool call to plain text content; never emit a malformed tool_use |
| Unparseable client request | Anthropic `invalid_request_error` (400) |

Invariant: **every** path out of the proxy is a spec-valid Anthropic
response. This is what "seamless" means operationally.

## 9. Observability

Structured per-request logs: tokens in/out, context-budget utilization,
stages applied, repairs triggered, retry counts, time-to-first-token.
`dump_prompts` writes each rendered backend prompt to disk for per-model
prompt tuning. `GET /stats` exposes session counters.

## 10. Testing strategy

- **Golden stage tests** — each pipeline stage: IR in → expected IR out.
- **Codec round-trip tests** — fixtures captured from a real Claude Code
  session via an early passthrough recording mode; decode → encode must be
  lossless for passthrough mode.
- **Fake backend** — scripted OpenAI-compatible server driving integration
  tests: streaming, malformed tool calls, mid-stream death, retry paths.
- **Live smoke script** — against a real local model (Ollama + Qwen first).
- TDD throughout.

## 11. Out of scope for v1

Image/document content blocks, web search & web fetch tool passthrough,
citations, prompt-caching emulation beyond accepting and ignoring
`cache_control`, LLM-based history summarization, multi-model routing,
authentication/multi-tenant concerns.
