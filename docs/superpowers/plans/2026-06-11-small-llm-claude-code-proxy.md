# Small-LLM Claude Code Proxy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A FastAPI proxy implementing the Anthropic Messages API that lets the real Claude Code CLI run effectively against local 14b–30b models served by any OpenAI-compatible backend.

**Architecture:** Anthropic codecs at the edges, a neutral IR in the middle, an ordered pipeline of IR→IR optimization stages, per-model-family profiles that render/parse backend traffic, and a relay loop that validates/repairs tool calls with bounded feedback retries. Spec: `docs/superpowers/specs/2026-06-11-small-llm-claude-code-proxy-design.md`.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, httpx (async SSE both directions), pydantic v2, json-repair, jsonschema, pytest + pytest-asyncio. Token counting is a heuristic counter behind a protocol (optional HF `tokenizers` upgrade later — deviation from spec noted: avoids model downloads in tests; interface is ready for the real tokenizer).

**Judgment deviations from spec (recorded here deliberately):**
1. Token counting defaults to a calibrated heuristic (~chars/3.6) behind a `TokenCounter` protocol; HF tokenizers is an optional drop-in.
2. Schema simplification = description trimming + schema-noise stripping; *dropping* optional params is deferred (risk > reward for core tools).
3. Degenerate-output handling: abort the stream cleanly (can't un-stream text); higher-temperature retry applies only when nothing has streamed yet.
4. Constrained decoding is applied on **repair retries** (force `tool_choice` + backend constraint param), not on first attempts — native tool calling is tried first.

---

## File map

```
pyproject.toml
harness.toml.example
src/harness/__init__.py
src/harness/__main__.py        # uvicorn entrypoint
src/harness/config.py          # Settings (TOML)
src/harness/ir.py              # IR dataclasses + stream events
src/harness/codec/anthropic_in.py
src/harness/codec/anthropic_out.py
src/harness/tokens/counter.py
src/harness/pipeline/base.py
src/harness/pipeline/system_prompt.py
src/harness/pipeline/tool_prune.py
src/harness/pipeline/tool_schema.py
src/harness/pipeline/history.py
src/harness/pipeline/fewshot.py
src/harness/profiles/base.py   # + TagSplitter for <think> handling
src/harness/profiles/registry.py  # qwen/deepseek_r1/devstral/gemma
src/harness/backends/base.py
src/harness/backends/openai_compat.py  # + vllm/llamacpp subclasses
src/harness/repair/toolcalls.py
src/harness/repair/degenerate.py
src/harness/relay.py           # orchestration loop
src/harness/server.py          # FastAPI app
tests/ (mirrors src), tests/fake_openai.py, tests/fixtures/
scripts/smoke.sh
README.md
```

---

### Task 1: Project scaffold + config

**Files:** Create `pyproject.toml`, `src/harness/__init__.py`, `src/harness/config.py`, `harness.toml.example`, `tests/test_config.py`

- [ ] **Step 1: failing test**

```python
# tests/test_config.py
from harness.config import Settings, load_settings

def test_defaults():
    s = Settings()
    assert s.server.port == 8484
    assert s.pipeline.system_prompt == "replace"
    assert s.pipeline.max_tools == 8

def test_load_toml(tmp_path):
    p = tmp_path / "harness.toml"
    p.write_text('[backend]\nmodel = "qwen2.5-coder:32b"\n[profile]\nname = "qwen"\n')
    s = load_settings(p)
    assert s.backend.model == "qwen2.5-coder:32b"
    assert s.profile.name == "qwen"
```

- [ ] **Step 2:** `pytest tests/test_config.py -v` → FAIL (module missing)
- [ ] **Step 3: implement**

```toml
# pyproject.toml
[project]
name = "ai-harness"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["fastapi>=0.115", "uvicorn[standard]>=0.30", "httpx>=0.27",
  "pydantic>=2.7", "json-repair>=0.30", "jsonschema>=4.22"]
[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.23"]
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
packages = ["src/harness"]
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

```python
# src/harness/config.py
import tomllib
from pathlib import Path
from pydantic import BaseModel

class ServerCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8484

class BackendCfg(BaseModel):
    kind: str = "openai"            # openai | vllm | llamacpp
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5-coder:14b"
    api_key: str = "local"

class ProfileCfg(BaseModel):
    name: str = "qwen"              # qwen | deepseek_r1 | devstral | gemma
    context_window: int = 32768

class PipelineCfg(BaseModel):
    system_prompt: str = "replace"  # replace | compress | passthrough
    tool_prune: bool = True
    max_tools: int = 8
    fewshot: bool = True
    repair_retries: int = 2
    recent_turns_protected: int = 4
    reasoning: str = "thinking"     # thinking | strip

class DebugCfg(BaseModel):
    dump_prompts: bool = False
    dump_dir: str = "debug_dumps"

class Settings(BaseModel):
    server: ServerCfg = ServerCfg()
    backend: BackendCfg = BackendCfg()
    profile: ProfileCfg = ProfileCfg()
    pipeline: PipelineCfg = PipelineCfg()
    debug: DebugCfg = DebugCfg()

def load_settings(path: str | Path | None = None) -> Settings:
    if path and Path(path).exists():
        return Settings.model_validate(tomllib.loads(Path(path).read_text()))
    return Settings()
```

`harness.toml.example`: the config block from spec §7. Create venv: `python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'`.

- [ ] **Step 4:** `pytest tests/test_config.py -v` → PASS
- [ ] **Step 5:** `git add -A && git commit -m "feat: project scaffold and config"`

### Task 2: IR

**Files:** Create `src/harness/ir.py`, `tests/test_ir.py`

- [ ] **Step 1: failing test**

```python
# tests/test_ir.py
from harness.ir import (Conversation, GenParams, TextPart, ToolCall,
                        ToolCallPart, ToolDef, ToolResultPart, Turn, Done)

def test_conversation_construction():
    conv = Conversation(
        system="be good",
        turns=(Turn("user", (TextPart("hi"),)),
               Turn("assistant", (ToolCallPart("t1", "Read", {"file_path": "/x"}),)),
               Turn("user", (ToolResultPart("t1", "contents"),))),
        tools=(ToolDef("Read", "Reads a file", {"type": "object"}, {"type": "object"}),),
        params=GenParams(max_tokens=4096))
    assert conv.turns[1].parts[0].name == "Read"

def test_frozen():
    import pytest
    with pytest.raises(Exception):
        TextPart("a").text = "b"
```

- [ ] **Step 2:** run → FAIL
- [ ] **Step 3: implement**

```python
# src/harness/ir.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, Union

@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    input_schema: dict[str, Any]      # what the model sees (maybe simplified)
    original_schema: dict[str, Any]   # what Claude Code sent; validation target

@dataclass(frozen=True)
class TextPart:
    text: str

@dataclass(frozen=True)
class ThinkingPart:
    text: str

@dataclass(frozen=True)
class ToolCallPart:
    id: str
    name: str
    arguments: dict[str, Any]

@dataclass(frozen=True)
class ToolResultPart:
    tool_call_id: str
    content: str
    is_error: bool = False

Part = Union[TextPart, ThinkingPart, ToolCallPart, ToolResultPart]

@dataclass(frozen=True)
class Turn:
    role: Literal["user", "assistant"]
    parts: tuple[Part, ...]

@dataclass(frozen=True)
class GenParams:
    max_tokens: int
    temperature: float | None = None
    stop_sequences: tuple[str, ...] = ()
    stream: bool = False

@dataclass(frozen=True)
class Conversation:
    system: str
    turns: tuple[Turn, ...]
    tools: tuple[ToolDef, ...]
    params: GenParams

# ---- stream events (relay/profile output) ----
@dataclass(frozen=True)
class TextDelta:
    text: str

@dataclass(frozen=True)
class ThinkingDelta:
    text: str

@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""           # original string when JSON parse failed

@dataclass(frozen=True)
class Done:
    stop_reason: str                  # end_turn | tool_use | max_tokens
    input_tokens: int = 0
    output_tokens: int = 0

IREvent = Union[TextDelta, ThinkingDelta, ToolCall, Done]
```

- [ ] **Step 4:** run → PASS
- [ ] **Step 5:** commit `feat: intermediate representation`

### Task 3: Anthropic request decode

**Files:** Create `src/harness/codec/__init__.py`, `src/harness/codec/anthropic_in.py`, `tests/test_anthropic_in.py`, `tests/fixtures/cc_request.json`

Fixture is a realistic Claude Code request: `system` as list of text blocks with `cache_control`, messages containing string content, text blocks, a `tool_use`/`tool_result` round trip (tool_result content as both string and block-list forms), `tools` with JSON schemas, `metadata.user_id`, `stop_sequences`, `stream: true`.

- [ ] **Step 1: failing tests** — decode fixture: system flattened to one string; tool_use → `ToolCallPart`; tool_result with list content flattened to text; `is_error` honored; tools → `ToolDef` with `original_schema == input_schema`; unknown/image blocks become `TextPart("[unsupported content omitted]")`.

```python
# tests/test_anthropic_in.py (core assertions)
import json
from pathlib import Path
from harness.codec.anthropic_in import decode
from harness.ir import TextPart, ToolCallPart, ToolResultPart

def fixture():
    return json.loads(Path("tests/fixtures/cc_request.json").read_text())

def test_decode_basics():
    conv = decode(fixture())
    assert "You are Claude Code" in conv.system
    assert conv.params.stream is True
    assert conv.tools[0].original_schema == conv.tools[0].input_schema

def test_decode_tool_roundtrip():
    conv = decode(fixture())
    calls = [p for t in conv.turns for p in t.parts if isinstance(p, ToolCallPart)]
    results = [p for t in conv.turns for p in t.parts if isinstance(p, ToolResultPart)]
    assert calls and results and calls[0].id == results[0].tool_call_id
```

- [ ] **Step 2:** run → FAIL
- [ ] **Step 3: implement** `decode(body: dict) -> Conversation`: flatten `system` (str or block list, join text blocks with `\n\n`); per message, normalize content str→`[{"type":"text",...}]`; map block types (`text`, `tool_use`, `tool_result`, `thinking`); flatten tool_result block-list content; collect tools; build `GenParams` from `max_tokens` (default 4096), `temperature`, `stop_sequences`, `stream`. Ignore `cache_control`, `metadata`, `tool_choice` (v1).
- [ ] **Step 4:** run → PASS
- [ ] **Step 5:** commit `feat: anthropic request decoder`

### Task 4: Anthropic response encode (SSE + JSON)

**Files:** Create `src/harness/codec/anthropic_out.py`, `tests/test_anthropic_out.py`

- [ ] **Step 1: failing tests** — drive `stream_sse(events, model, msg_id)` with an async generator yielding `[ThinkingDelta("hm"), TextDelta("Hello"), ToolCall("t1","Read",{"file_path":"/x"}), Done("tool_use", 10, 5)]` and assert the SSE text contains, in order: `message_start`, a `thinking` `content_block_start`, `thinking_delta`, `text` block + `text_delta "Hello"`, `tool_use` block start carrying `id/name` with `input_json_delta` whose `partial_json` parses to the args, `message_delta` with `"stop_reason": "tool_use"` and `"output_tokens": 5`, `message_stop`. Second test: `collect(events, ...)` returns a non-streaming Anthropic message dict with the same content blocks. Third test: `error_sse("overloaded_error", "backend down")` yields one well-formed `error` event.
- [ ] **Step 2:** run → FAIL
- [ ] **Step 3: implement**

```python
# src/harness/codec/anthropic_out.py (shape)
import json
from harness.ir import TextDelta, ThinkingDelta, ToolCall, Done

def _ev(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"

async def stream_sse(events, model: str, msg_id: str):
    yield _ev("message_start", {"type": "message_start", "message": {
        "id": msg_id, "type": "message", "role": "assistant", "model": model,
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0}}})
    yield _ev("ping", {"type": "ping"})
    idx, open_kind = -1, None   # open_kind: None | "text" | "thinking"
    # lazily open/close blocks; ToolCall closes any open block, then emits
    # content_block_start(tool_use id/name, input:{}) + one input_json_delta
    # (full json) + content_block_stop. Done closes open block, emits
    # message_delta {stop_reason, usage{output_tokens}} + message_stop.
    ...
```

(Full block-state machine implemented in this task; `collect()` reuses the same accumulation into a list of content blocks; map stop reasons verbatim — they already use Anthropic names in IR.)

- [ ] **Step 4:** run → PASS
- [ ] **Step 5:** commit `feat: anthropic SSE/JSON encoder`

### Task 5: Token counter

**Files:** Create `src/harness/tokens/__init__.py`, `src/harness/tokens/counter.py`, `tests/test_counter.py`

- [ ] **Step 1: failing test** — `HeuristicCounter().count_text("hello world " * 100)` within ±40% of 300; `count_conversation(conv)` > 0 and grows when turns are added; counts include tool schemas and system.
- [ ] **Step 2:** run → FAIL
- [ ] **Step 3: implement** — protocol `TokenCounter` with `count_text(str) -> int`; `HeuristicCounter`: `max(round(len(t)/3.6), len(t.split()))`. `count_conversation(conv, counter)`: system + every part's text/args-json + `json.dumps` of each tool schema + 8 tokens per message overhead.
- [ ] **Step 4:** run → PASS
- [ ] **Step 5:** commit `feat: token counting`

### Task 6: Pipeline base + system prompt rewrite

**Files:** Create `src/harness/pipeline/__init__.py`, `src/harness/pipeline/base.py`, `src/harness/pipeline/system_prompt.py`, `tests/test_system_prompt.py`

- [ ] **Step 1: failing tests** — given a synthetic 12k-char Claude Code system string (fingerprint line `You are Claude Code, Anthropic's official CLI for Claude.` + `# Tone and style` + `# Tool usage policy` sections + an `# Environment` section + a `# claudeMd` / `Contents of CLAUDE.md` section): `replace` mode output is < 4000 chars, contains the replacement contract rules, **retains the Environment and CLAUDE.md sections verbatim**, drops `# Tone and style`. Non-CC system prompts (no fingerprint) pass through `compress` (whitespace squeeze + dedupe) untouched in content. `passthrough` mode is identity.
- [ ] **Step 2:** run → FAIL
- [ ] **Step 3: implement**

```python
# src/harness/pipeline/base.py
from typing import Protocol
from harness.ir import Conversation
from harness.config import Settings

class Stage(Protocol):
    def apply(self, conv: Conversation, settings: Settings) -> Conversation: ...

def run_pipeline(conv, settings, stages):  # ordered list, honors toggles
    for st in stages:
        conv = st.apply(conv, settings)
    return conv
```

`system_prompt.py`: `FINGERPRINT = "You are Claude Code"`; split system on lines matching `^#{1,2} ` into (heading, body) sections; `KEEP = re.compile(r"environment|claude\.?md|memory|project|context", re.I)`; replacement prompt constant (the actual ~1.5k-token contract: act through tools; Read before Edit; exact-unique `old_string`; smallest change; Grep/Glob never guess paths; check Bash output; verify with tests; on tool error fix input, never repeat verbatim; ≤2 sentences before tool calls; stop when done). Rebuild: replacement + kept sections. `compress`: collapse 3+ blank lines, strip trailing spaces.

- [ ] **Step 4:** run → PASS
- [ ] **Step 5:** commit `feat: system prompt rewrite stage`

### Task 7: Tool pruning stage

**Files:** Create `src/harness/pipeline/tool_prune.py`, `tests/test_tool_prune.py`

- [ ] **Step 1: failing tests** — 15 tools in, `max_tools=8`: all core tools (`Read, Edit, Write, Bash, Grep, Glob, TodoWrite, Task`) kept; a non-core tool called within the last `recent_turns_protected` turns is kept (evicting lowest-priority core extra if over cap); tools absent from core and recent history are dropped; with `tool_prune=false` identity.
- [ ] **Step 2:** run → FAIL
- [ ] **Step 3: implement** — `CORE = ("Read","Edit","Write","Bash","Grep","Glob","TodoWrite","Task")`; recent names = ToolCallPart names in last K turns; keep order: recent ∪ core ∪ rest, truncated to max_tools.
- [ ] **Step 4:** run → PASS
- [ ] **Step 5:** commit `feat: tool pruning stage`

### Task 8: Schema simplification stage

**Files:** Create `src/harness/pipeline/tool_schema.py`, `tests/test_tool_schema.py`

- [ ] **Step 1: failing tests** — long tool description (>600 chars) trimmed to ≤300 ending at a sentence boundary; property descriptions ≤150 chars; `$schema`, `additionalProperties`, `title` keys stripped recursively; `anyOf` of `[X, {"type":"null"}]` flattened to `X`; `original_schema` untouched.
- [ ] **Step 2:** run → FAIL
- [ ] **Step 3: implement** — recursive `simplify(schema)`; sentence-boundary trim helper `trim(s, n)` (cut at last `. ` before n).
- [ ] **Step 4:** run → PASS
- [ ] **Step 5:** commit `feat: schema simplification stage`

### Task 9: History compaction stage

**Files:** Create `src/harness/pipeline/history.py`, `tests/test_history.py`

- [ ] **Step 1: failing tests** — build a conversation whose old tool results are huge; with a small `context_window`: (a) old ToolResultPart content > 1500 chars becomes head 800 + `\n…[elided by harness]…\n` + tail 300; (b) if still over budget, oldest assistant+tool_result turn *groups* are evicted together (no orphan tool_result whose call was evicted — assert pairing invariant); (c) last `recent_turns_protected` turns and system are byte-identical; (d) under-budget conversation is identity.
- [ ] **Step 2:** run → FAIL
- [ ] **Step 3: implement** — budget = `context_window - params.max_tokens - 1024` margin; loop: truncate eligible results oldest-first, recount, then evict turn-groups from the front (group = assistant turn + following user turn if it contains tool_results for its calls); insert a single `TextPart("[earlier conversation elided by harness]")` user turn marker at the eviction point.
- [ ] **Step 4:** run → PASS
- [ ] **Step 5:** commit `feat: history compaction stage`

### Task 10: Few-shot stage

**Files:** Create `src/harness/pipeline/fewshot.py`, `tests/test_fewshot.py`

- [ ] **Step 1: failing tests** — with `fewshot=true`, system gains a `## Tool call examples` section containing a Read→Edit→Bash sequence (correct param names: `file_path`, `old_string`/`new_string`, `command`); appended exactly once (idempotent on re-apply); off-toggle identity.
- [ ] **Step 2:** run → FAIL
- [ ] **Step 3: implement** — static example text showing three numbered exchanges (task → the exact tool + JSON args → brief result), ~30 lines.
- [ ] **Step 4:** run → PASS
- [ ] **Step 5:** commit `feat: few-shot example stage`

### Task 11: Model profiles (render + parse)

**Files:** Create `src/harness/profiles/__init__.py`, `src/harness/profiles/base.py`, `src/harness/profiles/registry.py`, `tests/test_profiles.py`

- [ ] **Step 1: failing tests**
  - base render: system message first; ToolResultPart → `{"role":"tool","tool_call_id",...}`; assistant ToolCallPart → `tool_calls[{id,type:"function",function:{name,arguments:json}}]`; tools rendered as OpenAI functions using **simplified** `input_schema`; payload has `stream:true`, `stream_options.include_usage`, `max_tokens`, `stop` when set.
  - base parse (feed scripted OpenAI chunk dicts): content deltas → `TextDelta`; `delta.reasoning_content` → `ThinkingDelta`; tool_call fragments accumulated by index → one `ToolCall` with parsed args at finish; `finish_reason "tool_calls"` → `Done("tool_use")`, `"length"` → `Done("max_tokens")`; usage chunk populates Done tokens; malformed args JSON → `ToolCall(arguments={}, raw_arguments=raw)`.
  - `TagSplitter("<think>", "</think>")`: tags split across chunk boundaries route text to thinking vs text correctly.
  - gemma render: no `system` role; system text prepended to first user message.
  - deepseek_r1 parse: `<think>` content → ThinkingDelta.
  - registry: `get_profile("qwen"|"deepseek_r1"|"devstral"|"gemma")` returns profiles; unknown name → ValueError.
- [ ] **Step 2:** run → FAIL
- [ ] **Step 3: implement**

```python
# src/harness/profiles/base.py (shape)
class Profile:
    name = "base"
    supports_system_role = True
    reasoning_tags: tuple[str, str] | None = None

    def render(self, conv: Conversation, model: str) -> dict: ...
    async def parse(self, chunks) -> AsyncIterator[IREvent]: ...

class TagSplitter:
    """Stateful router: text inside reasoning tags → thinking channel.
    Handles tags split across chunks via a holdback buffer of len(tag)-1."""
```

`registry.py`: `QwenProfile(Profile)` (defaults), `DeepseekR1Profile` (`reasoning_tags=("<think>","</think>")`), `DevstralProfile` (defaults), `GemmaProfile` (`supports_system_role=False`); `PROFILES` dict + `get_profile(name)`.

- [ ] **Step 4:** run → PASS
- [ ] **Step 5:** commit `feat: model profiles`

### Task 12: Backends

**Files:** Create `src/harness/backends/__init__.py`, `src/harness/backends/base.py`, `src/harness/backends/openai_compat.py`, `tests/fake_openai.py`, `tests/test_backends.py`

`tests/fake_openai.py`: tiny ASGI/FastAPI app, `POST /chat/completions`, behavior driven by a module-level `SCRIPT` list of chunk dicts; streams them as `data: {...}` SSE lines then `data: [DONE]`. Modes via special script entries: `{"_die_midstream": True}`, `{"_status": 500}`.

- [ ] **Step 1: failing tests** — using `httpx.AsyncClient(transport=ASGITransport(app))`: backend `stream(payload)` yields exactly the scripted parsed chunks; 500 → raises `BackendError`; mid-stream death → raises `BackendError` after partial chunks; `VllmBackend().apply_constraint(payload, schema)` adds `{"guided_json": schema, "tool_choice": "required"}` keys; `LlamaCppBackend` adds `{"json_schema": schema}`; base `OpenAIBackend.apply_constraint` is a no-op returning payload.
- [ ] **Step 2:** run → FAIL
- [ ] **Step 3: implement** — `base.py`: `class BackendError(Exception)`, `class Backend` (init with `BackendCfg` + injected `httpx.AsyncClient`; `constrained: bool = False`; `apply_constraint(payload, schema) -> payload`). `openai_compat.py`: POST `{base_url}/chat/completions`, `Authorization: Bearer`, iterate `aiter_lines()`, parse `data:` lines, stop at `[DONE]`; `VllmBackend`/`LlamaCppBackend` subclasses with `constrained=True` and their constraint keys. Factory `make_backend(cfg, client)`.
- [ ] **Step 4:** run → PASS
- [ ] **Step 5:** commit `feat: openai-compatible backends`

### Task 13: Repair (tool calls + degenerate output)

**Files:** Create `src/harness/repair/__init__.py`, `src/harness/repair/toolcalls.py`, `src/harness/repair/degenerate.py`, `tests/test_repair.py`

- [ ] **Step 1: failing tests**
  - valid call vs original schema → returned unchanged, no error.
  - `raw_arguments='{"file_path": "/x",}'` (trailing comma) → repaired via json_repair → valid.
  - missing required param → `(None, "validation error …file_path…")` string mentions the param.
  - unknown tool name → error mentioning available tool names.
  - degenerate: `DegenerateDetector.feed()` returns True when the tail repeats (`"abc " * 200` fed in chunks → True before 4000 chars); normal prose stays False.
- [ ] **Step 2:** run → FAIL
- [ ] **Step 3: implement** — `repair_toolcall(call: ToolCall, tools: tuple[ToolDef,...]) -> tuple[ToolCall | None, str | None]`: resolve tool by name → args = call.arguments or `json_repair.loads(raw_arguments)` → `jsonschema.validate(args, tool.original_schema)` → return repaired ToolCall or (None, error). `DegenerateDetector`: keep last 4000 chars; every 64 chars fed, check `for L in (24, 48, 96, 192): tail[-L:] == tail[-2L:-L] == tail[-3L:-2L]`.
- [ ] **Step 4:** run → PASS
- [ ] **Step 5:** commit `feat: tool call repair and degenerate detection`

### Task 14: Relay loop

**Files:** Create `src/harness/relay.py`, `tests/test_relay.py`

- [ ] **Step 1: failing tests** (scripted profile.parse outputs via fake backend scripts)
  - happy path: text + valid tool call stream through; events end with `Done("tool_use")`.
  - bad tool call then good: first response has invalid args → relay re-calls backend once with feedback turns appended (assert fake backend saw 2 requests and the 2nd request's messages include the validation error text); retry's text deltas suppressed; final valid ToolCall emitted.
  - retries exhausted → no ToolCall; a TextDelta containing the raw args and a `Done("end_turn")`.
  - constrained backend: on retry, second request payload contains `guided_json`.
  - degenerate text stream → relay stops yielding and emits `Done("end_turn")` early.
- [ ] **Step 2:** run → FAIL
- [ ] **Step 3: implement**

```python
# src/harness/relay.py (core control flow)
async def run(conv, profile, backend, settings) -> AsyncIterator[IREvent]:
    attempts, suppress_text = 0, False
    while True:
        payload = profile.render(conv, settings.backend.model)
        if attempts and backend.constrained and _last_schema is not None:
            payload = backend.apply_constraint(payload, _last_schema)
        retry_err = None
        detector = DegenerateDetector()
        async for ev in profile.parse(backend.stream(payload)):
            if isinstance(ev, (TextDelta, ThinkingDelta)):
                if suppress_text: continue
                if isinstance(ev, TextDelta) and detector.feed(ev.text):
                    yield Done("end_turn"); return
                yield ev
            elif isinstance(ev, ToolCall):
                fixed, err = repair_toolcall(ev, conv.tools)
                if fixed: yield fixed
                elif attempts < settings.pipeline.repair_retries:
                    retry_err, bad = err, ev; break
                else:
                    yield TextDelta(f"\n[invalid tool call: {err}]\n{ev.raw_arguments or ev.arguments}")
            else:  # Done
                yield ev; return
        if retry_err is None: return
        attempts += 1; suppress_text = True
        conv = _append_feedback(conv, bad, retry_err)   # assistant attempt + user error msg
```

`_append_feedback` adds the model's failed call as an assistant text turn and a user turn: `"Your tool call was invalid: {err}. Call {name} again with corrected JSON arguments matching its schema."` and records `_last_schema = tool.original_schema`.

- [ ] **Step 4:** run → PASS
- [ ] **Step 5:** commit `feat: relay loop with bounded repair retries`

### Task 15: Server (endpoints, error mapping, wiring)

**Files:** Create `src/harness/server.py`, `src/harness/__main__.py`, `tests/test_server.py`

- [ ] **Step 1: failing tests** (httpx ASGITransport against `create_app(settings, backend_client=fake)`)
  - `POST /v1/messages` stream=true: returns `text/event-stream`; full Anthropic event sequence for a scripted text+tool-call response; tool_use block has the validated input.
  - stream=false: complete Anthropic message JSON.
  - `POST /v1/messages/count_tokens` → `{"input_tokens": N}`, no backend call.
  - backend connect error → HTTP 529 body `{"type":"error","error":{"type":"overloaded_error",...}}`.
  - mid-stream backend death → stream ends with a valid `error` SSE event.
  - malformed request body → 400 `invalid_request_error`.
  - `GET /stats` → counters incremented.
  - pipeline smoke: request with CC fingerprint system + 15 tools → fake backend received: 1st message is rewritten system (< 4000 chars), ≤ 8 tools.
- [ ] **Step 2:** run → FAIL
- [ ] **Step 3: implement** — `create_app(settings, backend_client=None)`: builds profile, backend, stages `[SystemPromptStage(), ToolPruneStage(), ToolSchemaStage(), HistoryStage(), FewshotStage()]`; handler: decode → run_pipeline → relay.run → `stream_sse`/`collect`; wrap relay iteration in try/except BackendError to emit error SSE or 529; `msg_id = "msg_" + uuid4().hex`; stats counters on `app.state`; `dump_prompts` writes incoming request + rendered payload JSON to `debug.dump_dir`. `__main__.py`: argparse `--config`, uvicorn.run.
- [ ] **Step 4:** run → PASS, then full suite `pytest -q` → all green
- [ ] **Step 5:** commit `feat: server endpoints and wiring`

### Task 16: Smoke script + README

**Files:** Create `scripts/smoke.sh`, `README.md`, `harness.toml.example` (finalize)

- [ ] **Step 1:** `scripts/smoke.sh`: starts server with example config against `$OPENAI_BASE_URL` (default Ollama), curls a streaming `/v1/messages` request with one tool and prints events; exits non-zero if no `message_stop` seen.
- [ ] **Step 2:** README: what it is, quickstart (`pip install -e .`, edit `harness.toml`, run `python -m harness --config harness.toml`, point Claude Code: `ANTHROPIC_BASE_URL=http://localhost:8484 ANTHROPIC_API_KEY=local claude`), config reference table, per-model-family notes, limitations (spec §11).
- [ ] **Step 3:** run `bash scripts/smoke.sh` if a local model is available; otherwise verify script syntax with `bash -n`.
- [ ] **Step 4:** commit `docs: README and smoke script`

---

## Self-review notes

- **Spec coverage:** §2 endpoints → T15; §3 IR → T2; §4 components → T1–T15; §6 stages ①–⑤ → T6–T10; ⑥ → T12/T14; ⑦ → T13/T14; ⑧ → T11; streaming policy (buffer tool calls, stream text) → T4/T14; §7 config → T1; §8 errors → T15; §9 observability → T15 (stats + dump_prompts); §10 testing → fake backend T12, golden stage tests T6–T10, codec tests T3–T4. Recording mode is covered by `dump_prompts` capturing incoming requests.
- **Type consistency:** `ToolCall` event vs `ToolCallPart` turn-part are deliberately distinct; `repair_toolcall` signature used identically in T13/T14; `apply_constraint` defined T12, used T14.
- **No placeholders:** T4 and T14 show control-flow shape with the full state machines implemented in-task; all tests are concrete.
