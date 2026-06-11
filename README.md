# ai-harness

A Claude Code–compatible API proxy that maximizes the effectiveness of **small LLMs (14b–30b)**.

The real Claude Code CLI talks to this proxy as if it were the Anthropic API. The proxy
rewrites every request to fit how small models actually behave — then validates and
repairs the model's output so Claude Code always receives a spec-valid response.

```
┌────────────┐  Anthropic API   ┌────────────────────┐  OpenAI-compatible  ┌─────────────┐
│ Claude Code│ ───────────────▶ │  ai-harness        │ ──────────────────▶ │ 14b–30b LLM │
│ (real CLI) │ ◀─────────────── │  rewrite ▸ repair  │ ◀────────────────── │ Qwen, etc.  │
└────────────┘  SSE streaming   └────────────────────┘                     └─────────────┘
```

## What it does to each request

| Stage | Effect |
|---|---|
| System prompt rewrite | Claude Code's 10–20k-token prompt → ~1.5k-token small-model contract; environment/CLAUDE.md sections preserved verbatim |
| Tool pruning | ≤ 8 tools per turn (core set + recently used) |
| Schema simplification | Trimmed descriptions, de-noised JSON schemas; originals kept for validation |
| History compaction | Deterministic token-budget truncation/eviction; recent turns untouched |
| Few-shot injection | Canonical tool-call examples appended to the system prompt |
| Tool-call repair | JSON repair → schema validation → up to 2 feedback retries (schema-constrained on vLLM/llama.cpp) |
| Degenerate detection | Periodic-output detection aborts runaway streams cleanly |
| Reasoning handling | `<think>`/reasoning_content mapped to Anthropic thinking blocks (or stripped) |

## Fleet mode

Multiple inference servers become one API. Each `[[backends]]` entry has a model
profile and roles; the router gives every Claude Code conversation **session
affinity** (its growing prompt prefix stays hot in one server's KV cache),
sends haiku-class background calls to a `fast` backend, and fans sub-agent
traffic out to `subagent` backends with circuit-breaker failover.

Three cache layers:

| Layer | Mechanism | Saves |
|---|---|---|
| KV/prefix cache | session affinity + llama.cpp `cache_prompt` / vLLM `--enable-prefix-caching` | prompt recompute (TTFT) |
| Response cache | exact-match on rendered payload, fast-role requests | whole inference calls |
| Knowledge cache | `[memory]` per-project lessons injected next session | re-exploration tokens |

Cached prefix tokens are reported to Claude Code as `cache_read_input_tokens`,
so its native usage display works. Real per-backend numbers: `GET /dashboard`.

## Measured, not vibes

`evals/` runs real `claude -p` sessions over scripted tasks and produces an
A/B report (baseline vs full pipeline vs per-stage ablations):

```bash
.venv/bin/python evals/run.py --backend-url http://host:8000/v1 \
    --model <id> --profile qwen --kind vllm --configs baseline,full --trials 3
.venv/bin/python evals/report.py
```

Every request also feeds `[traces]` capture; `scripts/corpus.py` joins traces
with eval outcomes into an SFT-ready corpus for distilling smaller models.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
cp harness.toml.example harness.toml   # set backend.base_url + model + profile
.venv/bin/python -m harness --config harness.toml
```

Then point Claude Code at it:

```bash
ANTHROPIC_BASE_URL=http://localhost:8484 ANTHROPIC_API_KEY=local claude
```

## Configuration (`harness.toml`)

See `harness.toml.example` for every option. The essentials:

```toml
[backend]
kind = "openai"          # openai (Ollama/LM Studio/anything) | vllm | llamacpp
base_url = "http://localhost:11434/v1"
model = "qwen2.5-coder:32b"

[profile]
name = "qwen"            # qwen | deepseek_r1 | devstral | gemma
context_window = 32768

[pipeline]
system_prompt = "replace"  # replace | compress | passthrough
max_tools = 8
repair_retries = 2
reasoning = "thinking"     # thinking | strip
```

### Model family notes

- **qwen** — Qwen2.5-Coder-14B/32B, Qwen3-Coder-30B. Reference targets; default profile.
- **deepseek_r1** — R1-Distill-Qwen-14B/32B. `<think>` spans become Anthropic thinking blocks.
- **devstral** — Devstral Small 24B. Strong agentic tool use out of the box.
- **gemma** — Gemma 3. No system role; the proxy folds the system prompt into the first user message.
- `kind = "vllm"` / `"llamacpp"` additionally enable schema-constrained decoding on repair retries.

## Endpoints

- `POST /v1/messages` — streaming + non-streaming, the Claude Code main path
- `POST /v1/messages/count_tokens` — answered locally, no backend call
- `GET /stats` — request/error/token counters

## Debugging & tuning

Set `[debug] dump_prompts = true` to write every incoming Anthropic request and every
rendered backend payload to `debug_dumps/` — this is the capture corpus used for prompt
tuning and the downscale/distillation path (see `docs/superpowers/specs/2026-06-11-model-targeting-sow.md`).

## Development

```bash
.venv/bin/pytest -q          # full suite, no network needed (scripted fake backend)
bash scripts/smoke.sh        # live smoke against a real local model
```

## v1 limitations

No image/document blocks, no web tool passthrough, no citations, `cache_control` accepted
and ignored, no multi-tenant auth. See the design spec in `docs/superpowers/specs/`.
