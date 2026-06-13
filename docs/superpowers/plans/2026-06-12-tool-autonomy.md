# Tool Autonomy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The model sees a catalog of every available tool, gets full schemas for only ~8, and can call any catalogued tool — the harness surfaces the schema on demand. Fixes the bug where `Skill` and all MCP tools were invisible.

**Architecture:** Three changes. (1) `ToolPruneStage` selection priority becomes: tools called anywhere in history → tools named in the latest user message → CORE fills remaining slots; it also appends a byte-stable catalog of the full inventory to the system prompt and records the full inventory on `Conversation.all_tools`. (2) `relay.py` learns to surface a hidden tool's schema when the model calls it: valid args pass through at zero cost; invalid args swap the schema in and use the existing feedback-retry machinery. (3) A `tool_surfaced` metric flows to `logs/requests.jsonl` automatically via the existing `record.update(metrics)`.

**Tech Stack:** Python 3.12, pytest (run via `.venv/bin/python -m pytest`), dataclasses, no new dependencies.

**Constraints (from the platform spec, do not violate):**
- Prefix stability: the catalog must be byte-identical across all turns of a session; the surfaced tool set may only change at user-message boundaries or on a schema swap.
- Plain Python, small files, every behavior tested (self-maintainability).
- All tests must pass after every task: `.venv/bin/python -m pytest tests/ -q` → `... passed`.

**Run all commands from `/archive/ai-harness`.**

---

### Task 1: `Conversation.all_tools` carries the full inventory

The relay needs the full tool inventory to surface hidden schemas. The pruning stage is the last place that sees it, so it records it on the conversation.

**Files:**
- Modify: `src/harness/ir.py` (the `Conversation` dataclass, ~line 62)
- Modify: `src/harness/pipeline/tool_prune.py`
- Test: `tests/test_tool_prune.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_tool_prune.py`:

```python
def test_all_tools_records_full_inventory():
    # The relay surfaces hidden schemas from all_tools; pruning must
    # record the full inventory before cutting the surfaced set.
    out = ToolPruneStage().apply(conv(), Settings())
    assert out.all_tools == ALL_TOOLS
    assert len(out.tools) < len(out.all_tools)
```

- [ ] **Step 2: Run it, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tool_prune.py::test_all_tools_records_full_inventory -q`
Expected: FAIL with `TypeError` or `AttributeError` mentioning `all_tools` (the field does not exist yet).

- [ ] **Step 3: Add the field** — in `src/harness/ir.py`, change the `Conversation` dataclass to:

```python
@dataclass(frozen=True)
class Conversation:
    system: str
    turns: tuple[Turn, ...]
    tools: tuple[ToolDef, ...]
    params: GenParams
    # full client inventory; `tools` above is the surfaced subset whose
    # schemas the model sees. Empty when pruning is disabled.
    all_tools: tuple[ToolDef, ...] = ()
```

- [ ] **Step 4: Record it in the stage** — in `src/harness/pipeline/tool_prune.py`, change the last line of `ToolPruneStage.apply` from:

```python
        return replace(conv, tools=tuple(by_name[n] for n in keep))
```

to:

```python
        return replace(
            conv, tools=tuple(by_name[n] for n in keep), all_tools=conv.tools
        )
```

- [ ] **Step 5: Run the test, then the full suite**

Run: `.venv/bin/python -m pytest tests/test_tool_prune.py -q` → all pass.
Run: `.venv/bin/python -m pytest tests/ -q` → all pass (the new field has a default, so nothing else breaks).

- [ ] **Step 6: Commit**

```bash
git add src/harness/ir.py src/harness/pipeline/tool_prune.py tests/test_tool_prune.py
git commit -m "feat: Conversation.all_tools records full tool inventory

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: tools called anywhere in history stay surfaced

Today only the last 4 turns count, so a tool used 10 turns ago drops out and the tool list churns — every churn rewrites the prompt prefix and forces a full re-prefill. New rule: once called in surviving history, a tool stays, in first-call order (append-mostly = prefix-stable).

**Files:**
- Modify: `src/harness/pipeline/tool_prune.py`
- Test: `tests/test_tool_prune.py` (one test REPLACED deliberately, one added)

- [ ] **Step 1: Replace the obsolete test** — in `tests/test_tool_prune.py`, DELETE `test_old_usage_not_kept` (it asserts the old churn-prone behavior) and add in its place:

```python
def test_old_usage_stays_surfaced():
    # Once a tool is called it must stay surfaced for the whole session:
    # dropping it later changes the rendered tool list, which rewrites the
    # prompt prefix and forces a full KV re-prefill (20-60s at 60k tokens).
    s = Settings()
    old = (Turn("assistant", (ToolCallPart("t1", "WebFetch", {"url": "u"}),)),)
    filler = tuple(
        Turn("user", (TextPart(f"msg {i}"),))
        for i in range(s.pipeline.recent_turns_protected + 1)
    )
    out = ToolPruneStage().apply(conv(old + filler), s)
    assert "WebFetch" in {t.name for t in out.tools}


def test_called_tools_keep_first_call_order():
    # First-call order is append-mostly: new calls extend the list at a
    # stable position instead of reshuffling it.
    turns = (
        Turn("assistant", (ToolCallPart("t1", "NotebookEdit", {}),)),
        Turn("user", (TextPart("ok"),)),
        Turn("assistant", (ToolCallPart("t2", "WebFetch", {"url": "u"}),)),
        Turn("user", (TextPart("ok"),)),
    )
    out = ToolPruneStage().apply(conv(turns), Settings())
    names = [t.name for t in out.tools]
    assert names.index("NotebookEdit") < names.index("WebFetch")
```

- [ ] **Step 2: Run them, verify the new ones fail**

Run: `.venv/bin/python -m pytest tests/test_tool_prune.py -q`
Expected: `test_old_usage_stays_surfaced` FAILS (WebFetch missing); order test may pass or fail.

- [ ] **Step 3: Implement** — in `src/harness/pipeline/tool_prune.py`, replace the `recent` collection block inside `apply` (the `for turn in conv.turns[-settings.pipeline.recent_turns_protected:]` loop) with:

```python
        called: list[str] = []
        for turn in conv.turns:
            for part in turn.parts:
                if isinstance(part, ToolCallPart) and part.name not in called:
                    called.append(part.name)
```

and change the keep loop's source tuple from `(*recent, *CORE, *by_name)` to `(*called, *CORE, *by_name)`.

- [ ] **Step 4: Run the file's tests, then the full suite**

Run: `.venv/bin/python -m pytest tests/test_tool_prune.py -q` → all pass.
Run: `.venv/bin/python -m pytest tests/ -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add src/harness/pipeline/tool_prune.py tests/test_tool_prune.py
git commit -m "feat: history-called tools stay surfaced for prefix stability

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: tools named by the user get surfaced (deadlock fix)

"Use the github MCP" must surface `mcp__github__*` tools; mentioning a skill must surface the `Skill` tool. Matching is deterministic: word-boundary search of the latest real user message (the last user turn containing a `TextPart` — in agentic stretches the newest user turns hold only tool results, so this naturally finds the user's actual instruction and keeps the match stable until they speak again).

**Files:**
- Modify: `src/harness/pipeline/tool_prune.py`
- Test: `tests/test_tool_prune.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_tool_prune.py`:

```python
MCP_TOOLS = tuple(
    tool(n) for n in ("mcp__github__create_pr", "mcp__github__list_issues",
                      "mcp__slack__send_message", "Skill")
)


def conv_mcp(text: str) -> Conversation:
    return Conversation(
        "s",
        (Turn("user", (TextPart(text),)),),
        ALL_TOOLS + MCP_TOOLS,
        GenParams(max_tokens=100),
    )


def test_named_mcp_server_surfaces_its_tools():
    # Regression for the CORE deadlock: CORE is 8 names, max_tools is 8,
    # so an MCP tool could never be surfaced no matter what the user said.
    out = ToolPruneStage().apply(conv_mcp("use the github mcp to open a PR"), Settings())
    names = {t.name for t in out.tools}
    assert "mcp__github__create_pr" in names
    assert "mcp__github__list_issues" in names
    assert "mcp__slack__send_message" not in names
    assert len(out.tools) <= Settings().pipeline.max_tools


def test_exact_tool_name_surfaces_tool():
    out = ToolPruneStage().apply(conv_mcp("call mcp__slack__send_message please"), Settings())
    assert "mcp__slack__send_message" in {t.name for t in out.tools}


def test_skill_mention_surfaces_skill_tool():
    out = ToolPruneStage().apply(conv_mcp("run the brainstorming skill"), Settings())
    assert "Skill" in {t.name for t in out.tools}


def test_tool_results_do_not_trigger_matching():
    # File contents flowing back through tool results must not surface
    # tools; only the user's own words count.
    from harness.ir import ToolResultPart
    turns = (
        Turn("user", (TextPart("fix the bug"),)),
        Turn("assistant", (ToolCallPart("t1", "Read", {"file_path": "/x"}),)),
        Turn("user", (ToolResultPart("t1", "docs mention mcp__slack__send_message here"),)),
    )
    out = ToolPruneStage().apply(
        Conversation("s", turns, ALL_TOOLS + MCP_TOOLS, GenParams(max_tokens=100)),
        Settings(),
    )
    assert "mcp__slack__send_message" not in {t.name for t in out.tools}
```

- [ ] **Step 2: Run them, verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tool_prune.py -q`
Expected: the first three new tests FAIL (named tools missing); the fourth passes already.

- [ ] **Step 3: Implement the matcher** — in `src/harness/pipeline/tool_prune.py`, add `import re` and `TextPart` to the imports (`from harness.ir import Conversation, TextPart, ToolCallPart`), then add above the class:

```python
def _last_user_text(conv: Conversation) -> str:
    """The newest user turn that contains actual user words (TextPart).
    Tool-result-only turns are skipped so file contents can't trigger
    matches, and the match stays stable until the user speaks again."""
    for turn in reversed(conv.turns):
        if turn.role != "user":
            continue
        texts = [p.text for p in turn.parts if isinstance(p, TextPart)]
        if texts:
            return "\n".join(texts).lower()
    return ""


def _mentioned(word: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(word.lower())}s?\b", text) is not None


def _named_tools(conv: Conversation) -> list[str]:
    text = _last_user_text(conv)
    if not text:
        return []
    named: list[str] = []
    for t in conv.tools:
        if _mentioned(t.name, text):
            named.append(t.name)
        elif t.name.startswith("mcp__"):
            server = t.name.split("__")[1]
            if _mentioned(server, text):
                named.append(t.name)
    return named
```

then change the keep loop's source tuple from `(*called, *CORE, *by_name)` to `(*called, *_named_tools(conv), *CORE, *by_name)`.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_tool_prune.py -q` → all pass.
If `test_named_mcp_server_surfaces_its_tools` fails on the slack assertion, check `_mentioned`: "slack" must not be found in "use the github mcp to open a PR".

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q` → all pass.

- [ ] **Step 6: Commit**

```bash
git add src/harness/pipeline/tool_prune.py tests/test_tool_prune.py
git commit -m "feat: user-named tools and MCP servers surface into the tool set

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: the catalog — full inventory in the system prompt, byte-stable

One line per tool, appended to the system prompt. It lists the FULL inventory (not just hidden tools) so it never changes when the surfaced set changes — byte-identical across every turn of a session, living rent-free in the cached prefix.

**Files:**
- Modify: `src/harness/config.py` (`PipelineCfg`)
- Modify: `src/harness/pipeline/tool_prune.py`
- Modify: `harness.toml.example` (document the flag)
- Test: `tests/test_tool_prune.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_tool_prune.py`:

```python
def test_catalog_lists_full_inventory():
    out = ToolPruneStage().apply(conv(), Settings())
    assert "## Tool catalog" in out.system
    for t in ALL_TOOLS:
        assert f"- {t.name} " in out.system


def test_catalog_byte_stable_across_turns():
    # The catalog is part of the prompt prefix; if it varies between turns
    # the KV cache is invalidated every request. It must not depend on
    # which tools are currently surfaced.
    early = ToolPruneStage().apply(conv(), Settings())
    later_turns = (
        Turn("assistant", (ToolCallPart("t1", "WebFetch", {"url": "u"}),)),
        Turn("user", (TextPart("ok"),)),
    )
    later = ToolPruneStage().apply(conv(later_turns), Settings())
    catalog = lambda c: c.system.split("## Tool catalog", 1)[1]
    assert catalog(early) == catalog(later)


def test_catalog_disabled_by_flag():
    s = Settings()
    s.pipeline.tool_catalog = False
    out = ToolPruneStage().apply(conv(), s)
    assert "## Tool catalog" not in out.system


def test_catalog_summaries_are_short():
    long_desc = ToolDef("Verbose", "First sentence stays. " + "x" * 500,
                        {"type": "object"}, {"type": "object"})
    c = Conversation("s", (), ALL_TOOLS + (long_desc,), GenParams(max_tokens=100))
    out = ToolPruneStage().apply(c, Settings())
    line = next(l for l in out.system.splitlines() if l.startswith("- Verbose"))
    assert len(line) <= 100
    assert "First sentence stays" in line
```

- [ ] **Step 2: Run them, verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tool_prune.py -q`
Expected: all four new tests FAIL (`## Tool catalog` absent; `tool_catalog` flag missing raises on the flag test).

- [ ] **Step 3: Add the config flag** — in `src/harness/config.py`, add one line to `PipelineCfg` after `tool_prune: bool = True`:

```python
    tool_catalog: bool = True  # list the full tool inventory in the system prompt
```

- [ ] **Step 4: Implement the catalog** — in `src/harness/pipeline/tool_prune.py`, add above the class:

```python
CATALOG_HEADER = (
    "## Tool catalog\n"
    "You may call ANY tool below by name, even if its full schema is not "
    "provided; the schema will be supplied when you use it."
)


def _summary(description: str) -> str:
    first = description.strip().split("\n", 1)[0]
    first = first.split(". ", 1)[0].rstrip(".")
    return first[:80]


def _catalog(tools: tuple) -> str:
    lines = [f"- {t.name} — {_summary(t.description)}" for t in tools]
    return CATALOG_HEADER + "\n" + "\n".join(lines)
```

and change the stage's return to:

```python
        system = conv.system
        if settings.pipeline.tool_catalog:
            system = system + "\n\n" + _catalog(conv.tools)
        return replace(
            conv,
            tools=tuple(by_name[n] for n in keep),
            all_tools=conv.tools,
            system=system,
        )
```

(The catalog is built from `conv.tools` — the full inventory, since this runs pre-prune — and depends on nothing else, which is what makes it byte-stable.)

- [ ] **Step 5: Document the flag** — in `harness.toml.example`, under the `[pipeline]` section, add:

```toml
tool_catalog = true   # list every tool (1 line each) so the model can call hidden ones
```

- [ ] **Step 6: Run the tests, then the full suite**

Run: `.venv/bin/python -m pytest tests/test_tool_prune.py -q` → all pass.
Run: `.venv/bin/python -m pytest tests/ -q` → all pass.

- [ ] **Step 7: Commit**

```bash
git add src/harness/config.py src/harness/pipeline/tool_prune.py harness.toml.example tests/test_tool_prune.py
git commit -m "feat: byte-stable full-inventory tool catalog in the system prompt

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: relay surfaces a hidden tool when the call is already valid

The model read the catalog and called a hidden tool with correct arguments. That must succeed on the spot — no retry, no extra tokens.

**Files:**
- Modify: `src/harness/relay.py`
- Test: `tests/test_relay.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_relay.py`:

```python
WEB_SCHEMA = {
    "type": "object",
    "properties": {"url": {"type": "string"}},
    "required": ["url"],
}


def conv_with_hidden_tool() -> Conversation:
    read = ToolDef("Read", "reads", READ_SCHEMA, READ_SCHEMA)
    web = ToolDef("WebFetch", "fetches a url", WEB_SCHEMA, WEB_SCHEMA)
    return Conversation(
        "sys",
        (Turn("user", (TextPart("fetch x"),)),),
        (read,),                      # only Read is surfaced
        GenParams(max_tokens=512, stream=True),
        all_tools=(read, web),        # WebFetch is catalog-only
    )


async def test_hidden_tool_valid_call_passes_through():
    # Model called a catalogued-but-unsurfaced tool with valid args:
    # zero-cost path, no retry round-trip.
    fake = FakeOpenAI()
    fake.push([tool_chunk("c1", "WebFetch", '{"url": "https://x"}'),
               finish_chunk("tool_calls")])
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [e async for e in run(conv_with_hidden_tool(), get_profile("qwen"),
                                backend, Settings(), metrics=metrics)]
    assert any(isinstance(e, ToolCall) and e.name == "WebFetch" for e in evs)
    assert len(fake.requests) == 1
    assert metrics["tool_surfaced"] == 1
```

- [ ] **Step 2: Run it, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_relay.py::test_hidden_tool_valid_call_passes_through -q`
Expected: FAIL — no ToolCall event (today an unknown tool goes down the feedback/invalid path) or KeyError on `tool_surfaced`.

- [ ] **Step 3: Implement** — in `src/harness/relay.py`:

(a) add a helper above `run` (and add `Conversation` — already imported):

```python
def _surface_tool(conv: Conversation, name: str) -> Conversation | None:
    """The model called a catalogued tool whose schema is not surfaced.
    Returns conv with the real ToolDef added (so validation, feedback,
    and constrained retries all see it), or None if the name is unknown."""
    if any(t.name == name for t in conv.tools):
        return None
    tool = next((t for t in conv.all_tools if t.name == name), None)
    if tool is None:
        return None
    return replace(conv, tools=conv.tools + (tool,))
```

(b) in `run`, add to the metrics defaults block:

```python
    m.setdefault("tool_surfaced", 0)
```

(c) in the `elif isinstance(ev, ToolCall):` branch, change:

```python
                fixed, error = repair_toolcall(ev, conv.tools)
```

to:

```python
                fixed, error = repair_toolcall(ev, conv.tools)
                if fixed is None:
                    surfaced = _surface_tool(conv, ev.name)
                    if surfaced is not None:
                        conv = surfaced
                        m["tool_surfaced"] += 1
                        fixed, error = repair_toolcall(ev, conv.tools)
```

- [ ] **Step 4: Run the test, then the full relay file**

Run: `.venv/bin/python -m pytest tests/test_relay.py -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add src/harness/relay.py tests/test_relay.py
git commit -m "feat: relay surfaces hidden tool schema on a valid catalog call

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: relay swaps the schema in and retries when the call is invalid

The model reached for a hidden tool but got the arguments wrong. Swap the real schema into the tool set and let the EXISTING feedback-retry machinery (including grammar-constrained retry on vLLM) fix it. No new retry logic.

**Files:**
- Modify: none expected — Task 5's code already produces this behavior; this task PROVES it with tests.
- Test: `tests/test_relay.py`

- [ ] **Step 1: Write the tests** — append to `tests/test_relay.py`:

```python
async def test_hidden_tool_invalid_call_swaps_schema_and_retries():
    fake = FakeOpenAI()
    fake.push([tool_chunk("c1", "WebFetch", '{"address": "x"}'),   # wrong param
               finish_chunk("tool_calls")])
    fake.push([tool_chunk("c2", "WebFetch", '{"url": "https://x"}'),
               finish_chunk("tool_calls")])
    fake.push([finish_chunk("stop")])  # safety
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [e async for e in run(conv_with_hidden_tool(), get_profile("qwen"),
                                backend, Settings(), metrics=metrics)]
    assert len(fake.requests) == 2
    # the retry request must offer the WebFetch schema
    retry_tools = [t["function"]["name"] for t in fake.requests[1].get("tools", [])]
    assert "WebFetch" in retry_tools
    # and the valid second call is emitted
    assert any(isinstance(e, ToolCall) and e.arguments == {"url": "https://x"} for e in evs)
    assert metrics["tool_surfaced"] == 1


async def test_truly_unknown_tool_still_fails_with_feedback():
    # A tool in neither the surfaced set nor the catalog keeps today's
    # behavior: feedback retry, then degrade to text.
    fake = FakeOpenAI()
    bad = [tool_chunk("c1", "Nonexistent", '{"a": 1}'), finish_chunk("tool_calls")]
    fake.push(bad)
    fake.push(bad)
    fake.push(bad)
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [e async for e in run(conv_with_hidden_tool(), get_profile("qwen"),
                                backend, Settings(), metrics=metrics)]
    assert not any(isinstance(e, ToolCall) for e in evs)
    assert metrics["tool_surfaced"] == 0
```

- [ ] **Step 2: Run them**

Run: `.venv/bin/python -m pytest tests/test_relay.py -q`
Expected: both PASS if Task 5 was implemented correctly (the swap feeds the existing retry path). If the first FAILS on `retry_tools`, verify the profile renders `conv.tools` into the payload `tools` field and that `_surface_tool` ran before `_append_feedback`; the `conv` variable must be reassigned (`conv = surfaced`) so the retry render includes the swapped schema.

- [ ] **Step 3: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q` → all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_relay.py
git commit -m "test: schema swap feeds the existing feedback-retry machinery

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: end-to-end pipeline test, then close out

Prove the loop converges across requests: a hidden tool that got called appears in the next request's history, so selection keeps it surfaced without any relay help.

**Files:**
- Test: `tests/test_tool_prune.py`

- [ ] **Step 1: Write the test** — append to `tests/test_tool_prune.py`:

```python
def test_called_hidden_tool_stays_surfaced_next_request():
    # Convergence: after the relay surfaces a tool and the model uses it,
    # the NEXT request's history contains that call, so selection keeps
    # the tool surfaced without the relay's help.
    from harness.ir import ToolResultPart
    turns = (
        Turn("user", (TextPart("send the report to slack"),)),
        Turn("assistant", (ToolCallPart("t1", "mcp__slack__send_message", {"text": "hi"}),)),
        Turn("user", (ToolResultPart("t1", "sent"),)),
        Turn("assistant", (TextPart("done"),)),
        Turn("user", (TextPart("now also post it to the channel"),)),
    )
    out = ToolPruneStage().apply(
        Conversation("s", turns, ALL_TOOLS + MCP_TOOLS, GenParams(max_tokens=100)),
        Settings(),
    )
    assert "mcp__slack__send_message" in {t.name for t in out.tools}
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests/test_tool_prune.py -q`
Expected: PASS immediately (Task 2 built this); if it fails, the called-priority loop is not scanning all turns.

- [ ] **Step 3: Full suite + line count check**

Run: `.venv/bin/python -m pytest tests/ -q` → all pass.
Run: `wc -l src/harness/pipeline/tool_prune.py` → must stay under ~120 lines (self-maintainability; if larger, move the matcher helpers to `src/harness/pipeline/tool_match.py` and import them).

- [ ] **Step 4: Commit**

```bash
git add tests/test_tool_prune.py
git commit -m "test: surfaced tools converge to sticky across requests

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Verification after all tasks

1. `.venv/bin/python -m pytest tests/ -q` — everything passes.
2. Manual smoke: start the harness, ask Claude Code (pointed at it) to "use the <your MCP server> to …" and watch `logs/requests.jsonl` for `"tool_surfaced": 1` entries.
3. Observability: `grep -c tool_surfaced logs/requests.jsonl` grows over normal use; zero forever means the matcher or catalog is not reaching the model.

## Explicitly out of scope (next plans)

- `tool-discovery` eval task family → belongs to the **eval expansion** plan (build-order #2), which must first read `evals/run.py` to follow the existing task format (`prompt.txt` + `check.sh` + `repo_template/`).
- Per-model catalog sizing (4b vs 27b) → eval expansion will measure whether the catalog distracts the smallest models.
