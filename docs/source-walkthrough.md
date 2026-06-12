# Source Walkthrough

This guide explains how `ai-harness` is put together so a new engineer can make
changes without having to reverse-engineer the request path from scratch.

The short version: the server accepts Anthropic Messages API requests from
Claude Code, converts them into an internal representation, runs pipeline
stages that make the request easier for smaller models, renders the result as
OpenAI chat completions, streams from a backend, repairs tool calls when needed,
and converts the result back into Anthropic output.

## Mental Model

There are four boundaries in the codebase:

1. **Client boundary**: Anthropic request and response shapes.
2. **Internal boundary**: neutral conversation and stream event dataclasses.
3. **Backend boundary**: OpenAI-compatible chat completions payloads and SSE
   chunks.
4. **Operational boundary**: routing, cache, logs, traces, stats, and dashboard.

Most feature work should happen on one side of one boundary. If a change touches
several boundaries, slow down and add tests around the conversion points.

## Request Lifecycle

The main request path starts in `src/harness/server.py`.

1. `POST /v1/messages` receives the Claude Code request body.
2. `harness.codec.anthropic_in.decode()` converts the Anthropic body into the
   IR `Conversation`.
3. `Router.pick()` chooses a backend before the pipeline runs. This matters
   because history compaction uses the selected backend's context window.
4. `run_pipeline()` applies system prompt, tool, schema, history, few-shot, and
   memory stages.
5. The selected profile renders the IR as an OpenAI chat completions payload.
6. If response caching applies and the rendered payload has been seen before,
   cached IR events are replayed instead of calling the backend.
7. Otherwise `relay.run()` streams backend chunks, parses them into IR events,
   validates tool calls, and performs bounded repair retries.
8. `stream_sse()` turns IR events into Anthropic SSE for streaming clients.
   `collect()` turns the same events into a non-streaming Anthropic JSON body.
9. The server records stats, request logs, traces, cache entries, and memory
   notes after the response path finishes.

The important invariant is that the rest of the code does not work directly
with Anthropic or OpenAI wire data. Wire formats are decoded at the edges,
converted to IR, then encoded again at the edges.

## Internal Representation

`src/harness/ir.py` defines the central dataclasses:

- `Conversation`: system prompt, turns, tools, and generation parameters.
- `Turn`: one user or assistant turn containing typed parts.
- `ToolDef`: the tool schema sent to the model plus the original schema used
  for validation.
- `TextPart`, `ThinkingPart`, `ToolCallPart`, `ToolResultPart`: content inside
  prior conversation turns.
- `TextDelta`, `ThinkingDelta`, `ToolCall`, `Done`: streamed output events.

If you add support for a new content type, start here, then update both codecs
and the relevant profile render/parse logic.

## Anthropic Codecs

`src/harness/codec/anthropic_in.py` decodes client requests.

Key behavior:

- Text, thinking, tool use, and tool result blocks are preserved in typed IR.
- Unsupported content becomes a placeholder text part instead of crashing.
- Tool definitions keep the original schema. Later pipeline stages may simplify
  the model-facing schema, but repair still validates against the original.

`src/harness/codec/anthropic_out.py` encodes IR events back to Anthropic output.

Key behavior:

- `stream_sse()` emits a valid Anthropic SSE sequence, including block start,
  delta, block stop, message delta, and message stop events.
- `collect()` builds the equivalent non-streaming message body.
- Usage maps backend cached prompt tokens into Anthropic
  `cache_read_input_tokens`.
- Error helpers produce Anthropic-shaped errors so Claude Code gets predictable
  failures.

If Claude Code rejects output, inspect this module first.

## Backend Profiles

`src/harness/profiles/base.py` owns the default OpenAI-compatible render and
parse behavior.

Rendering:

- System prompts become OpenAI `system` messages when the profile supports the
  system role.
- Profiles without system-role support fold the system prompt into the first
  user message.
- Assistant tool calls become OpenAI `tool_calls`.
- Tool results become OpenAI `tool` messages.
- Backend requests always use `stream = true` so the parser can collect text,
  tool calls, and usage consistently. Non-streaming client responses are built
  by collecting IR events on the harness side.

Parsing:

- OpenAI streaming deltas become `TextDelta`, `ThinkingDelta`, and `ToolCall`
  events.
- Streaming tool-call fragments are accumulated by tool-call index.
- Malformed JSON arguments are preserved in `raw_arguments` so the repair layer
  can try to fix them.
- Finish reasons are mapped back to Anthropic-style stop reasons.
- Cached-token usage is read from OpenAI usage details when present, and from
  llama.cpp timing data when available.

`src/harness/profiles/registry.py` registers model-family profiles:

- `qwen`: default behavior.
- `deepseek_r1`: splits `<think>...</think>` into thinking events.
- `devstral`: default behavior with a distinct profile name.
- `gemma`: folds system prompts into the first user message.

Add a profile when a model family needs different render or parse behavior. Do
not put model-family quirks in the server or relay unless they are truly
backend-wide.

## Pipeline Stages

The pipeline protocol is in `src/harness/pipeline/base.py`. Each stage receives
a `Conversation` and returns a new `Conversation`.

Current stages run in this order:

1. `SystemPromptStage`: replaces or compresses Claude Code's system prompt.
   Recognized Claude Code prompts are rebuilt as a smaller agent contract plus
   selected project context sections.
2. `ToolPruneStage`: limits visible tools to recent tools, core Claude Code
   tools, and then any remaining tools up to `max_tools`.
3. `ToolSchemaStage`: trims verbose descriptions and removes schema noise while
   preserving `original_schema` for validation.
4. `HistoryStage`: enforces context budget by truncating old tool results first
   and evicting old turn groups second. Recent turns are protected.
5. `FewshotStage`: appends concrete tool-use examples to the system prompt.
6. `MemoryStage`: optionally injects durable project facts from prior sessions.

Pipeline stages should be deterministic. Rewriting old context differently on
every request hurts backend prefix caching.

## Routing And Backend Pool

`src/harness/backends/pool.py` wraps backend configuration and runtime state.
Single-backend config is normalized into a one-entry pool, so the rest of the
server always uses the same pool path.

Each `PooledBackend` tracks:

- backend config and profile;
- supported roles such as `main`, `fast`, and `subagent`;
- in-flight count;
- circuit-breaker cooldown state;
- request, error, token, latency, cache, and KV-residency counters.

`src/harness/router.py` chooses a backend for each request:

- Session affinity wins first, so a conversation stays on the same backend and
  benefits from prefix/KV cache reuse.
- Requests whose model name contains `haiku` use the `fast` role.
- Main requests prefer `main` backends, with overflow to `subagent` backends
  when all main backends are busy.
- Down backends are skipped unless every candidate is down.

`POST /admin/reload` re-reads backend definitions and updates surviving backend
objects in place, preserving counters and in-flight streams.

## Backend Clients

`src/harness/backends/openai_compat.py` contains the downstream clients.

- `OpenAIBackend` streams from `/chat/completions` and yields parsed SSE JSON
  chunks.
- `VllmBackend` adds `guided_json` during repair retries.
- `LlamaCppBackend` sets `cache_prompt = true` and adds `json_schema` during
  repair retries.

Backend classes should stay thin. Family-specific prompt or parser behavior
belongs in profiles; transport and constraint flags belong in backend classes.

## Relay And Tool Repair

`src/harness/relay.py` coordinates the model call after the pipeline has run.

The relay loop:

- renders the current conversation with the selected profile;
- streams and parses backend chunks into IR events;
- passes text and thinking through unless reasoning is configured to be
  stripped;
- detects repetitive output and terminates with a `Done` event;
- validates each tool call against the original Claude Code schema;
- locally repairs JSON when possible;
- feeds invalid tool calls back to the model for a bounded number of retries;
- applies backend schema constraints on repair retries when the backend supports
  them;
- always terminates the stream with `Done`.

`src/harness/repair/toolcalls.py` is the validation and JSON-repair layer.
`src/harness/repair/degenerate.py` is a small repetition detector for runaway
streams.

When debugging bad tool behavior, inspect this sequence:

1. rendered backend payload dump;
2. raw backend stream chunk;
3. profile parser output;
4. repair result;
5. Anthropic output encoding.

## Cache, Logs, Traces, And Memory

`src/harness/cache.py` provides an in-memory exact-match response cache for
selected roles. The cache key is the rendered backend payload with stream flags
removed. Hits replay previously collected IR events.

`src/harness/log.py` writes one compact JSONL record per request when
`[log] requests_path` is configured. On startup, `server.py` can replay this log
to rehydrate aggregate stats.

`src/harness/traces.py` records rendered payloads, IR events, and metrics for
evaluation and corpus generation. The eval runner can tag traces with
`HARNESS_TRACE_TAG`.

`src/harness/memory.py` implements the optional project memory layer. It records
session tails, waits until a session is idle, asks a fast backend to extract
durable project facts, and injects those facts into future sessions for the
same project.

These systems are best-effort. They should never break the serving path.

## Metrics And Dashboard

`GET /stats` is assembled in `src/harness/server.py`.

It reports:

- global request, error, input, output, and cached-token counts;
- per-backend roles, in-flight counts, circuit-breaker state, latency
  percentiles, cache-hit percentages, and KV-write estimates;
- response-cache hit and miss counts.

For vLLM and llama.cpp, the server tries to read backend `/metrics` gauges for
KV occupancy. For llama.cpp, if gauges are unavailable, it estimates occupancy
from `/slots` and recent resident sessions. The dashboard at
`src/harness/static/dashboard.html` renders this JSON.

## Testing Map

The tests are organized by module behavior:

- `tests/test_anthropic_in.py` and `tests/test_anthropic_out.py`: codec shape.
- `tests/test_profiles.py`: profile render and parse behavior.
- `tests/test_system_prompt.py`, `tests/test_tool_prune.py`,
  `tests/test_tool_schema.py`, `tests/test_history.py`, `tests/test_fewshot.py`:
  pipeline behavior.
- `tests/test_repair.py` and `tests/test_relay.py`: validation, repair, and
  relay loop behavior.
- `tests/test_pool_router.py` and `tests/test_backends.py`: routing, pool, and
  backend transport behavior.
- `tests/test_server.py`, `tests/test_server_logging.py`,
  `tests/test_response_cache.py`, `tests/test_cache_mechanics.py`: integrated
  server behavior.
- `tests/test_memory.py`, `tests/test_traces.py`, `tests/test_evals.py`: support
  systems.

Run all tests with:

```bash
.venv/bin/pytest -q
```

## Where To Start For Common Changes

- **Add a new model family**: create a profile in `profiles/registry.py`, add
  render or parse overrides in `profiles/base.py` if needed, and add tests in
  `tests/test_profiles.py`.
- **Support a new backend constraint mechanism**: add a backend subclass in
  `backends/openai_compat.py`, register its kind, and test `apply_constraint`.
- **Change prompt rewriting**: edit `pipeline/system_prompt.py` and update
  `tests/test_system_prompt.py`.
- **Change tool repair behavior**: edit `repair/toolcalls.py` or `relay.py` and
  update `tests/test_repair.py` plus `tests/test_relay.py`.
- **Change routing policy**: edit `router.py` and `backends/pool.py`, then
  update `tests/test_pool_router.py`.
- **Add a new operational metric**: record it in `server.py`, include it in
  `/stats`, and update dashboard/tests that assert the stats shape.

## Development Rules Of Thumb

- Keep wire-format knowledge in codecs and profiles.
- Keep backend transport details in backend classes.
- Keep request-shaping behavior in pipeline stages.
- Preserve `original_schema` whenever tool schemas are simplified.
- Make cache-affecting transformations deterministic.
- Treat logs, traces, memory, and dashboard updates as non-critical support
  paths; they should not make normal inference fail.
- Add tests at the boundary where data changes shape.
