# ai-harness

`ai-harness` is an Anthropic Messages API shim for running Claude Code against
OpenAI-compatible local or private inference servers. Claude Code talks to this
service as if it were the Anthropic API; the harness rewrites the request for a
smaller coding model, routes it to one or more backends, repairs malformed tool
calls, and streams a spec-shaped Anthropic response back to Claude Code.

The target use case is getting useful agentic coding behavior out of 14B-30B
class models and small backend fleets, while preserving Claude Code's normal
CLI workflow.

```text
Claude Code
  -> Anthropic /v1/messages
  -> ai-harness request pipeline, router, repair loop, metrics
  -> OpenAI-compatible /chat/completions backend
  -> vLLM, llama.cpp server, Ollama, LM Studio, OpenRouter, or similar
```

## Problems This Project Solves

Claude Code is built around Anthropic's API shape, model behavior, and tool-use
reliability. Local and smaller open models expose a different API, have tighter
context windows, and are much less forgiving when handed Claude Code's full
prompt and full tool surface. `ai-harness` is the compatibility and control
plane between those worlds.

The project is attempting to solve these concrete problems:

- **API mismatch**: Claude Code emits Anthropic Messages requests, while most
  self-hosted inference stacks expose OpenAI chat completions. The harness
  translates both directions, including streaming responses.
- **Prompt scale mismatch**: Claude Code's system prompt can be large enough to
  waste most of a small model's useful context. The harness rewrites or
  compresses it into a smaller operational contract.
- **Tool-use fragility**: Small models often emit malformed JSON, invalid tool
  names, or arguments that do not satisfy the original schema. The harness
  reduces the active tool surface and repairs tool calls before Claude Code sees
  them.
- **Context budget pressure**: Smaller and mixed-size backends need predictable
  history compaction based on the actual routed model's context window.
- **Fleet utilization**: A single Claude Code session can involve main-agent,
  background, and subagent traffic. The harness routes those roles across a
  backend fleet while keeping conversation affinity for prefix-cache reuse.
- **Operational visibility**: Local inference deployments need quick feedback on
  request volume, errors, latency, token flow, cache hits, and backend health.
  The harness exposes `/stats` and `/dashboard` for that loop.
- **Experimentation**: Prompt rewrites, repair behavior, caching, memory, and
  model choices need to be measurable. The repo includes eval tasks, request
  logging, traces, and corpus tooling for iteration.

## What It Does

Each Claude Code request is decoded into an internal conversation format, passed
through a pipeline, rendered as OpenAI chat completions, streamed from the model,
then encoded back to Anthropic SSE or JSON.

Core behavior:

- Replaces or compresses Claude Code's large system prompt into a smaller model
  contract while preserving project and environment sections.
- Prunes tools to a small active set so weaker models are less likely to choose
  invalid tools.
- Simplifies JSON schemas for the model while validating against the original
  tool contracts.
- Compacts history to fit the selected backend context window.
- Injects few-shot examples for canonical tool-call formatting.
- Repairs malformed tool calls with JSON repair, schema validation, and bounded
  retry feedback.
- Maps model reasoning output into Anthropic thinking blocks or strips it,
  depending on profile settings.
- Tracks request counts, token usage, latency, cache hits, backend health, and
  KV-cache occupancy where the backend exposes it.

## Repository Layout

- `src/harness/server.py` - FastAPI application and Anthropic-compatible routes.
- `src/harness/pipeline/` - prompt rewrite, tool pruning, schema cleanup,
  history compaction, few-shot injection, and memory injection.
- `src/harness/backends/` - OpenAI-compatible backend clients and fleet state.
- `src/harness/profiles/` - model-family render and parse behavior.
- `src/harness/repair/` - tool-call repair and degenerate output detection.
- `src/harness/static/dashboard.html` - browser dashboard at `/dashboard`.
- `evals/` - scripted Claude Code evaluation tasks and reports.
- `scripts/smoke.sh` - live smoke test against a real OpenAI-compatible backend.

For a detailed source-code map, read `docs/source-walkthrough.md`.

## Requirements

- Python 3.11 or newer.
- One or more OpenAI-compatible chat-completions servers.
- Claude Code configured to use this service as its Anthropic base URL.

Supported backend kinds:

- `openai` - generic OpenAI-compatible APIs, including Ollama, LM Studio, and
  hosted compatible endpoints.
- `vllm` - OpenAI-compatible vLLM with guided JSON support for repair retries.
- `llamacpp` - llama.cpp server, with `cache_prompt` enabled by the harness and
  optional `/metrics` and `/slots` polling for KV stats.

Supported model profiles:

- `qwen` - Qwen coder models; this is the default and main reference profile.
- `deepseek_r1` - DeepSeek R1 distill models with `<think>` handling.
- `devstral` - Devstral Small style tool-use behavior.
- `gemma` - Gemma models, including system-prompt folding for no-system-role
  backends.

## Local Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp harness.toml.example harness.toml
```

Edit `harness.toml` for your backend. A minimal single-backend config looks
like this:

```toml
[server]
host = "127.0.0.1"
port = 8484

[backend]
kind = "openai"
base_url = "http://localhost:11434/v1"
model = "qwen2.5-coder:32b"
api_key = "local"

[profile]
name = "qwen"
context_window = 32768
```

Start the harness:

```bash
.venv/bin/python -m harness --config harness.toml
```

Point Claude Code at it:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8484 ANTHROPIC_API_KEY=local claude
```

The API key is passed through only as a client-facing Anthropic key here. The
downstream backend key is configured separately as `backend.api_key` or per
`[[backends]]` entry.

## Docker Quickstart

Build and run with Docker Compose:

```bash
# Edit docker/harness.toml for your backend and model.
# The default points at an OpenAI-compatible backend on the Docker host.
docker compose up --build
```

Then point Claude Code at the container:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8484 ANTHROPIC_API_KEY=local claude
```

The compose file mounts `./docker/harness.toml` at `/config/harness.toml` and
stores request logs in `./logs` when logging is enabled. On Linux, the compose
file maps `host.docker.internal` to the Docker host so the container can reach a
model server running directly on the host.

## Fleet Deployment

Fleet mode is the intended deployment shape when you have multiple model
servers. Define one `[[backends]]` table per server and assign roles:

```toml
[server]
host = "0.0.0.0"
port = 8484

[[backends]]
name = "main-qwen"
kind = "vllm"
base_url = "http://gpu-a:8000/v1"
model = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
api_key = "local"
profile = "qwen"
context_window = 131072
roles = ["main"]

[[backends]]
name = "fast-qwen"
kind = "llamacpp"
base_url = "http://gpu-b:8080/v1"
model = "qwen2.5-coder-14b"
api_key = "local"
profile = "qwen"
context_window = 32768
roles = ["fast", "subagent"]

[cache]
enabled = true
ttl_s = 600
max_entries = 256
roles = ["fast"]

[log]
requests_path = "logs/requests.jsonl"
```

The router keeps each Claude Code conversation on a stable backend when
possible so the backend's prefix or KV cache stays warm. Requests that look like
Claude Haiku/background traffic use the `fast` role, and subagent traffic can be
spread across `subagent` backends. Backends that error are cooled down
temporarily by the circuit breaker.

For vLLM, start the model with prefix caching enabled when available. For
llama.cpp server, enable metrics if you want live KV usage in `/stats` and
`/dashboard`.

## Process Deployment

The harness is a normal long-running ASGI process launched through its module
entrypoint:

```bash
cd /opt/ai-harness
.venv/bin/python -m harness --config /etc/ai-harness/harness.toml
```

Recommended production shape:

- Run the harness on the same trusted LAN or host as Claude Code users and model
  backends.
- Bind `server.host = "127.0.0.1"` for single-user local use, or `0.0.0.0`
  behind a firewall or reverse proxy for LAN use.
- Put TLS, authentication, and access control in front of the harness if it is
  reachable outside a trusted machine. The app itself is not a multi-tenant auth
  boundary.
- Keep backend model servers private; expose only the harness endpoint to Claude
  Code clients.
- Enable `[log] requests_path` if you want stats to survive restarts and need a
  request audit trail.
- Keep `[debug] dump_prompts = false` outside prompt-tuning sessions because
  dumps may contain source code, paths, and conversation content.

A minimal systemd unit:

```ini
[Unit]
Description=ai-harness Claude Code proxy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/ai-harness
ExecStart=/opt/ai-harness/.venv/bin/python -m harness --config /etc/ai-harness/harness.toml
Restart=on-failure
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Then configure clients:

```bash
export ANTHROPIC_BASE_URL=http://harness-host:8484
export ANTHROPIC_API_KEY=local
claude
```

## Operations

Useful endpoints:

- `POST /v1/messages` - Anthropic Messages API surface used by Claude Code.
- `POST /v1/messages/count_tokens` - local heuristic token counting.
- `GET /stats` - JSON counters, backend health, latency, cache, and KV metrics.
- `GET /dashboard` - browser dashboard for the same operational data.
- `POST /admin/reload` - reloads backend definitions from `harness.toml`.

`/admin/reload` updates only backend fleet configuration. Pipeline, cache,
debug, memory, logging, and server bind changes require a process restart.

## Evaluation And Tuning

Run the unit tests:

```bash
.venv/bin/pytest -q
```

Run a live smoke test against a local OpenAI-compatible backend:

```bash
OPENAI_BASE_URL=http://localhost:11434/v1 MODEL=qwen2.5-coder:14b bash scripts/smoke.sh
```

Run scripted Claude Code evals:

```bash
.venv/bin/python evals/run.py --backend-url http://host:8000/v1 \
  --model <model-id> --profile qwen --kind vllm --configs baseline,full --trials 3
.venv/bin/python evals/report.py
```

Set `[traces] enabled = true` to capture rendered backend payloads and model
events. `scripts/corpus.py` can join traces with eval outcomes into a corpus for
fine-tuning or distillation experiments.

## Limitations

- No image or document block support.
- No web tool passthrough or citation support.
- `cache_control` is accepted but ignored.
- Authentication and tenancy are expected to be handled outside the process.
- Token counting is heuristic and designed for routing and budgeting, not exact
  billing.

## License

GPL-2.0-only. See `LICENSE`.
