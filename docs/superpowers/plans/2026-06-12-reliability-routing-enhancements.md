# Reliability & Routing Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the measured top defects from 1,386 proxied requests: phantom empty-name tool calls (96 invalid / 144 retries on qwen27), main-role overload (81% of traffic on one backend), task loss under long-session eviction, context-free eviction markers, and the unstarted SFT data flywheel.

**Architecture:** Five independent, individually-shippable changes: a parser flush guard in the qwen/openai profile, fingerprint-based role detection in the router, an eviction anchor + structured digest in the history stage, and config flips for traces/dumps. Each lands with a failing test first.

**Tech Stack:** Python 3.12, pytest, FastAPI test transport (existing `tests/fake_openai.py`).

**Evidence anchors (measured 2026-06-12):**
- Phantom slot reproduced live against qwen27 (vLLM 0.19.1-patched): parallel calls stream `idx 1: id=…, name=null, args=""` and nothing else; flushed as `ToolCall(id, "", {})` → `unknown tool ''`.
- Request distribution: qwen27 1121, qwen35 145, gemma31 120 — router maps everything non-haiku to `main`.
- Yesterday's compaction bug class: task message evicted; first-turn pinning is the regression guard.

---

### Task 1: Drop phantom tool-call slots at parse flush

**Files:**
- Modify: `src/harness/profiles/base.py:198-204` (the flush loop in `parse`)
- Test: `tests/test_profiles.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_profiles.py` (reuse that file's existing chunk-builder helpers if present; otherwise this standalone version):

```python
async def test_phantom_tool_slot_dropped():
    # vLLM 0.19.x parallel-call bug: a slot streams an id with name=null and
    # empty args, then never fills. It must not surface as a ToolCall.
    from harness.profiles.registry import get_profile
    from harness.ir import ToolCall

    chunks = [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "a", "function": {"name": "Read", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "a", "function": {"name": None, "arguments": "{\"file_path\": \"/x\"}"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 1, "id": "b", "function": {"name": None, "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 2, "id": "c", "function": {"name": "Read", "arguments": "{\"file_path\": \"/y\"}"}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]

    async def stream():
        for c in chunks:
            yield c

    events = [ev async for ev in get_profile("qwen").parse(stream())]
    calls = [ev for ev in events if isinstance(ev, ToolCall)]
    assert [c.name for c in calls] == ["Read", "Read"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_profiles.py::test_phantom_tool_slot_dropped -q`
Expected: FAIL — three ToolCalls yielded, one with name `""`.

- [ ] **Step 3: Minimal implementation**

In `src/harness/profiles/base.py`, flush loop:

```python
        for slot in calls.values():
            raw = slot["arguments"]
            if not slot["name"] and not raw.strip():
                # vLLM ≤0.19 parallel-call phantom: id-only slot, no name, no
                # args — carries zero information, dropping it avoids a retry.
                continue
            try:
                args = json.loads(raw) if raw else {}
                yield ToolCall(slot["id"], slot["name"], args)
            except json.JSONDecodeError:
                yield ToolCall(slot["id"], slot["name"], {}, raw_arguments=raw)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_profiles.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/harness/profiles/base.py tests/test_profiles.py
git commit -m "fix: drop phantom empty tool-call slots from vLLM parallel-call streams"
```

### Task 2: Fingerprint-based role detection in the router

**Files:**
- Modify: `src/harness/router.py` (add `request_role`, use in `pick`)
- Modify: `src/harness/server.py` (the `role = "fast" if "haiku" …` line → `request_role(body)`; import it)
- Test: `tests/test_pool_router.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pool_router.py` (adapt to its existing fixture style):

```python
def test_request_role_main_cli():
    from harness.router import request_role
    body = {"model": "claude-opus-4-8",
            "system": "You are Claude Code, Anthropic's official CLI for Claude."}
    assert request_role(body) == "main"


def test_request_role_subagent_sdk():
    from harness.router import request_role
    body = {"model": "claude-opus-4-8",
            "system": [{"type": "text", "text": "You are a Claude agent, built on Anthropic's Claude Agent SDK."}]}
    assert request_role(body) == "subagent"


def test_request_role_haiku_fast():
    from harness.router import request_role
    assert request_role({"model": "claude-haiku-4-5"}) == "fast"


def test_request_role_unknown_defaults_main():
    from harness.router import request_role
    assert request_role({"model": "claude-opus-4-8", "system": "custom"}) == "main"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_pool_router.py -q`
Expected: FAIL — `request_role` not defined.

- [ ] **Step 3: Minimal implementation**

In `src/harness/router.py` after `session_key`:

```python
MAIN_FINGERPRINT = "You are Claude Code, Anthropic's official CLI"
SUBAGENT_MARKERS = ("Claude Agent SDK", "You are an agent for Claude Code")


def request_role(body: dict) -> str:
    """fast: haiku-class. subagent: Task/SDK agent fingerprints.
    main: the interactive CLI loop, and the safe default for unknowns."""
    if "haiku" in (body.get("model") or ""):
        return "fast"
    system = _flatten(body.get("system") or "")[:KEY_BASIS_CHARS]
    if MAIN_FINGERPRINT in system:
        return "main"
    if any(marker in system for marker in SUBAGENT_MARKERS):
        return "subagent"
    return "main"
```

In `Router.pick`, replace the role line and mirror the busy-overflow for subagent:

```python
        role = request_role(body)
        candidates = self.pool.with_role(role)
        if role in ("main", "subagent"):
            # overflow: if every backend for the role is busy (or none up),
            # widen to the other agentic role
            other = "subagent" if role == "main" else "main"
            if not candidates or min(b.in_flight for b in candidates) > 0:
                candidates = candidates + self.pool.with_role(other)
```

In `src/harness/server.py`: import `request_role` alongside `Router, session_key`; replace the inline `role = "fast" if "haiku" in (body.get("model") or "") else "main"` with `role = request_role(body)`.

- [ ] **Step 4: Full suite**

Run: `.venv/bin/python -m pytest tests/ -q` — all pass (watch the fleet-routing tests near `_fleet_toml` in `test_server.py`; if one asserts subagent prompts route to `main`, that assertion encodes the old bug and may be updated with a comment).

- [ ] **Step 5: Commit**

```bash
git add src/harness/router.py src/harness/server.py tests/test_pool_router.py
git commit -m "feat: route Task/SDK subagent requests to the subagent role"
```

### Task 3: Pin the first user turn during eviction

**Files:**
- Modify: `src/harness/pipeline/history.py` (pass 2)
- Test: `tests/test_history.py`

- [ ] **Step 1: Write the failing test**

```python
def test_eviction_pins_first_user_turn():
    # The opening user turn carries the task; ghost-task behavior returns if
    # it is ever evicted. It must survive any amount of compaction.
    conv = big_session(20, 8000)
    out = HistoryStage().apply(conv, small_settings(4000))
    first_texts = [
        p.text for p in out.turns[0].parts if isinstance(p, TextPart)
    ] + [p.text for p in out.turns[1].parts if isinstance(p, TextPart)]
    assert any("fix the bug" in t for t in first_texts)
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest tests/test_history.py::test_eviction_pins_first_user_turn -q`
Expected: FAIL — turn 0 is the eviction marker, "fix the bug" evicted.

- [ ] **Step 3: Minimal implementation**

In `HistoryStage.apply`, before pass 2 splits `head` into groups, peel off an anchor:

```python
        # pass 2: evict turn-groups from the front, but never the anchor —
        # the opening user turn that states the task.
        anchor: tuple[Turn, ...] = ()
        if head and head[0].role == "user" and any(
            isinstance(p, TextPart) for p in head[0].parts
        ):
            anchor, head = head[:1], head[1:]
        groups = _groups(head)
        evicted = False
        while groups and count_conversation(conv, self.counter) > target:
            groups.pop(0)
            evicted = True
            kept = tuple(t for g in groups for t in g)
            marker = (EVICT_MARKER,) if evicted else ()
            conv = replace(conv, turns=anchor + marker + kept + tail)
        return conv
```

- [ ] **Step 4: Verify pass + suite**

Run: `.venv/bin/python -m pytest tests/test_history.py tests/test_server.py -q`

- [ ] **Step 5: Commit**

```bash
git add src/harness/pipeline/history.py tests/test_history.py
git commit -m "fix: pin the opening task turn through history eviction"
```

### Task 4: Structured eviction digest instead of a bare marker

**Files:**
- Modify: `src/harness/pipeline/history.py` (replace constant `EVICT_MARKER` with `_digest(evicted_groups)`)
- Test: `tests/test_history.py`

Note: v1 is deterministic (tool names + counts), not an LLM summary — zero
latency, zero dependency on the fast backend being warm, and byte-stable
between compaction events so the KV prefix is only invalidated when eviction
itself changes. An LLM summary can layer on later behind config.

- [ ] **Step 1: Write the failing test**

```python
def test_eviction_digest_names_tools():
    conv = big_session(20, 8000)
    out = HistoryStage().apply(conv, small_settings(4000))
    marker_texts = [
        p.text for t in out.turns for p in t.parts
        if isinstance(p, TextPart) and "elided" in p.text
    ]
    assert marker_texts, "digest marker missing"
    assert "Read" in marker_texts[0]          # names the tools used
    assert "turns" in marker_texts[0]          # says how much was cut
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest tests/test_history.py::test_eviction_digest_names_tools -q`
Expected: FAIL — marker is the bare "[earlier conversation elided by harness]".

- [ ] **Step 3: Minimal implementation**

Replace the `EVICT_MARKER` constant with a builder, and accumulate evicted groups in the pass-2 loop:

```python
DIGEST_MAX_TOOLS = 8


def _digest(evicted: list[tuple[Turn, ...]]) -> Turn:
    n_turns = sum(len(g) for g in evicted)
    tools: list[str] = []
    for g in evicted:
        for t in g:
            for p in t.parts:
                if isinstance(p, ToolCallPart) and p.name not in tools:
                    tools.append(p.name)
    used = ", ".join(tools[:DIGEST_MAX_TOOLS]) or "no tools"
    text = (
        f"[{n_turns} earlier turns elided by harness; tools used: {used}. "
        "Results of that work appear in later turns.]"
    )
    return Turn("user", (TextPart(text),))
```

(import `ToolCallPart` from `harness.ir`.) In the loop, collect `dropped.append(groups.pop(0))` and build `marker = (_digest(dropped),)`.

- [ ] **Step 4: Verify pass + full suite**

Run: `.venv/bin/python -m pytest tests/ -q` — `test_eviction_keeps_pairing` asserts on the word "elided", which the digest retains.

- [ ] **Step 5: Commit**

```bash
git add src/harness/pipeline/history.py tests/test_history.py
git commit -m "feat: eviction digest names elided tool activity"
```

### Task 5: Flip the flywheel — traces on, prompt dumps off

**Files:**
- Modify: `harness.toml`

- [ ] **Step 1: Edit config**

```toml
[debug]
dump_prompts = false

[traces]
enabled = true
```

- [ ] **Step 2: Smoke-check the corpus builder against captured traces**

Run after some live traffic: `.venv/bin/python scripts/corpus.py --help` (verify CLI loads; corpus build itself needs accumulated traces).

- [ ] **Step 3: Restart service and verify**

```bash
kill <pid>; nohup .venv/bin/python -m harness --config harness.toml >> logs/harness.out 2>&1 &
curl -s localhost:8484/stats | python3 -m json.tool | head
```

- [ ] **Step 4: Commit**

```bash
git add harness.toml
git commit -m "chore: enable trace capture, disable prompt dumps"
```

## Self-Review

- Spec coverage: item 1→Task 2, item 2 (re-scoped on evidence)→Task 1, item 3→Task 3, item 4 (deterministic v1)→Task 4, item 5+ops→Task 5. The runtime task-survival *guard* is covered by Task 3's pinning plus yesterday's server-level test; a separate eval-suite invariant is deferred until the eval phase starts.
- Placeholders: none; all steps carry code/commands.
- Type consistency: `request_role` used identically in router/server; `_digest` returns `Turn` matching `EVICT_MARKER`'s prior type.
