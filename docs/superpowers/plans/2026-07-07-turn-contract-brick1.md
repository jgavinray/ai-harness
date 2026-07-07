# Turn Contract Brick 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A relay turn can never end as a silent give-up (prose-only or empty `end_turn` after the model attempted to act) and never reports zero tokens when work happened — closing the "give-up death" failure class from the 2026-07-07 consistency spec.

**Architecture:** All changes live in the relay loop (`src/harness/relay.py`). Three additions: (1) usage accumulates across retry attempts so every terminal `Done` carries real totals; (2) a prose-only/empty `end_turn` on a turn where the model attempted a tool call (or the action state requires one) is fed back like a repair, bounded by the existing `repair_retries` budget; (3) when that budget is exhausted the turn ends with an honest structured failure message, never a bare empty `end_turn`. New counters (`contract_feedback`, `gave_up_honestly`, guard `give_up`) flow into `requests.jsonl` automatically via `server.py:903` (`record.update(metrics)`).

**Tech Stack:** Python 3, pytest (async tests via anyio plugin already configured), FakeOpenAI scripted backend in `tests/fake_openai.py`.

## Global Constraints

- Run tests with `.venv/bin/pytest` from `/archive/ai-harness` (spec: tests are the maintainer's safety net).
- Prefix stability: no new per-turn prompt injection; feedback turns append only (same pattern as existing `_append_*_feedback` helpers).
- Feedback wording must not contain the words verify/check/build/compile/lint or "run tests" — those words flip `action_state` into `verify` state on the next attempt (see `_has_verify_intent` in `src/harness/action_state.py:57`).
- The working tree has uncommitted changes in `harness.toml`, `docker-compose.yml`, `docker/harness.toml` — these are the user's local config experiments. NEVER stage or commit those three files.
- Plain Python, no metaprogramming (umbrella spec self-maintainability rule).
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 0: Baseline green + commit the in-flight working-tree code

The user's uncommitted diff in `src/harness/` and `tests/` (vLLM grammar subset, loop-signature normalization, critic runtime triggers, tool_schema propertyNames) is prerequisite work this brick builds on. Commit it as its own commit so brick-1 commits stay clean.

**Files:**
- Commit (already modified, do not edit): `src/harness/backends/openai_compat.py`, `src/harness/critic.py`, `src/harness/pipeline/tool_schema.py`, `src/harness/relay.py`, `src/harness/review.py`, `src/harness/server.py`, `tests/test_backends.py`, `tests/test_critic.py`, `tests/test_tool_schema.py`

**Interfaces:**
- Produces: a clean git state where `src/` and `tests/` have no uncommitted changes (config files stay dirty by design).

- [ ] **Step 1: Run the full test suite on the current tree**

Run: `cd /archive/ai-harness && .venv/bin/pytest -q`
Expected: all tests pass (0 failures). If anything fails, STOP and report — the in-flight diff is not green and the user must decide.

- [ ] **Step 2: Commit the in-flight source changes only**

```bash
cd /archive/ai-harness
git add src/harness/backends/openai_compat.py src/harness/critic.py \
  src/harness/pipeline/tool_schema.py src/harness/relay.py \
  src/harness/review.py src/harness/server.py \
  tests/test_backends.py tests/test_critic.py tests/test_tool_schema.py
git commit -m "fix(relay): grammar-subset constraints, loop-signature normalization, critic runtime triggers

vLLM guided_json now receives only the schema subset its grammar compiler
accepts; Bash loop detection normalizes whitespace; critic triggers extend
the review manager; tool_schema strips propertyNames.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 3: Verify only config files remain dirty**

Run: `git -C /archive/ai-harness status --porcelain`
Expected output contains ONLY `harness.toml`, `docker-compose.yml`, `docker/harness.toml`, and untracked `docs/nvidia-gpu-fan-control.md`, `recovery/`.

---

### Task 1: Usage accumulation across relay attempts

Every retry attempt whose stream reaches its `Done` chunk carries usage that is currently thrown away; fabricated terminal `Done` events report zeros (the observed 0-token dead sessions in `evals/results/final-audit-rerun3`). Accumulate input/output tokens across attempts and make every terminal `Done` carry the totals.

**Files:**
- Modify: `src/harness/relay.py` (the `run()` function: init vars near line 250, degenerate abort near line 331, Done branch near line 435, stream-ended fallback near line 532)
- Test: `tests/test_relay.py`

**Interfaces:**
- Consumes: `Done(stop_reason, input_tokens=0, output_tokens=0, cached_tokens=0)` from `src/harness/ir.py:96`; `finish_chunk(reason, prompt_tokens=10, completion_tokens=5)` from `tests/fake_openai.py`.
- Produces: every `Done` yielded by `relay.run()` carries accumulated `input_tokens`/`output_tokens` summed over all attempts whose `Done` chunk was parsed, and `cached_tokens` as the max seen. Tasks 2 and 3 use the local variables `total_input`, `total_output`, `total_cached`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_relay.py` (uses the existing `conv()` fixture whose user text "read x" puts the action state in tool-required `inspect`, so a prose-only first attempt is retried by the existing action-state block — two attempts, two parsed `Done` chunks):

```python
async def test_usage_accumulates_across_feedback_attempts():
    fake = FakeOpenAI()
    fake.push([text_chunk("thinking out loud"), finish_chunk("stop")])
    fake.push([tool_chunk("c1", "Read", '{"file_path": "/x"}'), finish_chunk("tool_calls")])
    evs = await collect_events(fake)
    done = evs[-1]
    assert done.stop_reason == "tool_use"
    # both attempts' usage chunks (10/5 each) are summed, not dropped
    assert done.input_tokens == 20
    assert done.output_tokens == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_relay.py::test_usage_accumulates_across_feedback_attempts -q`
Expected: FAIL with `assert 10 == 20` (only the last attempt's usage survives today).

- [ ] **Step 3: Implement accumulation in relay.py**

In `src/harness/relay.py`, inside `run()`, directly after the line `require_tool_after_invalid_skill = False` (near line 250), add:

```python
    total_input = 0
    total_output = 0
    total_cached = 0
```

Replace the degenerate-abort block (near line 328):

```python
                if isinstance(ev, TextDelta) and detector.feed(ev.text):
                    m["degenerate_aborts"] += 1
                    yield TextDelta("\n[output truncated: repetition detected]")
                    yield Done("end_turn")
                    return
```

with:

```python
                if isinstance(ev, TextDelta) and detector.feed(ev.text):
                    m["degenerate_aborts"] += 1
                    yield TextDelta("\n[output truncated: repetition detected]")
                    yield Done("end_turn", total_input, total_output, total_cached)
                    return
```

Replace the whole `else:  # Done` branch (near lines 435–458):

```python
            else:  # Done
                if buffered_text and not emitted_valid_call and ev.stop_reason != "tool_use":
                    guarded_done = guard_done_claim(conv, "".join(buffered_text), settings)
                    if guarded_done is not None and attempts < settings.pipeline.repair_retries:
                        break
                    if effective_requires_tool and attempts < settings.pipeline.repair_retries:
                        action_state_feedback = (action_state.name, [tool.name for tool in payload_conv.tools])
                        break
                    for text in buffered_text:
                        yield TextDelta(text)
                if not emitted_valid_call and ev.stop_reason == "tool_use":
                    # every call this turn failed validation and retries are gone
                    yield Done("end_turn", ev.input_tokens, ev.output_tokens)
                else:
                    if (
                        require_tool_after_invalid_skill
                        and not emitted_valid_call
                        and ev.stop_reason != "tool_use"
                        and attempts < settings.pipeline.repair_retries
                    ):
                        tool_required_after_invalid_skill = True
                        break
                    yield ev
                return
```

with:

```python
            else:  # Done
                total_input += ev.input_tokens
                total_output += ev.output_tokens
                total_cached = max(total_cached, ev.cached_tokens)
                if buffered_text and not emitted_valid_call and ev.stop_reason != "tool_use":
                    guarded_done = guard_done_claim(conv, "".join(buffered_text), settings)
                    if guarded_done is not None and attempts < settings.pipeline.repair_retries:
                        break
                    if effective_requires_tool and attempts < settings.pipeline.repair_retries:
                        action_state_feedback = (action_state.name, [tool.name for tool in payload_conv.tools])
                        break
                    for text in buffered_text:
                        yield TextDelta(text)
                if not emitted_valid_call and ev.stop_reason == "tool_use":
                    # every call this turn failed validation and retries are gone
                    yield Done("end_turn", total_input, total_output, total_cached)
                else:
                    if (
                        require_tool_after_invalid_skill
                        and not emitted_valid_call
                        and ev.stop_reason != "tool_use"
                        and attempts < settings.pipeline.repair_retries
                    ):
                        tool_required_after_invalid_skill = True
                        break
                    yield Done(ev.stop_reason, total_input, total_output, total_cached)
                return
```

Replace the stream-ended fallback (near line 530):

```python
        if bad_call is None:
            # stream ended without a Done (backend quirk); close the turn
            yield Done("tool_use" if emitted_valid_call else "end_turn")
            return
```

with:

```python
        if bad_call is None:
            # stream ended without a Done (backend quirk); close the turn
            yield Done(
                "tool_use" if emitted_valid_call else "end_turn",
                total_input, total_output, total_cached,
            )
            return
```

- [ ] **Step 4: Run the new test and the full relay suite**

Run: `.venv/bin/pytest tests/test_relay.py -q`
Expected: all pass, including `test_usage_accumulates_across_feedback_attempts`.

- [ ] **Step 5: Commit**

```bash
cd /archive/ai-harness
git add src/harness/relay.py tests/test_relay.py
git commit -m "fix(relay): accumulate usage across retry attempts

Terminal Done events now carry summed input/output tokens and max cached
tokens over all attempts whose usage chunk arrived, instead of dropping
prior attempts and fabricating zero-token Dones (the 0-token dead-session
signature in final-audit-rerun3).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Give-up contract — feed back prose-only/empty end_turn after an attempted action

The observed give-up death: the model attempts a tool call, it fails validation, the retry produces suppressed prose and `end_turn`, and the relay hands Claude Code an empty final turn — session over in 4 s. Contract: if the model attempted any tool call this turn (or the action state requires one) and the turn is ending `end_turn` with no valid call and no visible text, feed it back like a repair.

**Files:**
- Modify: `src/harness/relay.py` (new feedback helper near line 154; per-run and per-attempt state; Done branch; feedback dispatch after the stream loop near line 514)
- Test: `tests/test_relay.py`

**Interfaces:**
- Consumes: `total_input`/`total_output`/`total_cached` from Task 1; `increment_guard` from `harness.guards` (already imported in relay.py).
- Produces: metrics keys `contract_feedback` (int) and guard fire `give_up`; local flag `attempted_action: bool` and per-attempt flag `give_up_feedback: bool` that Task 3 extends with the exhaustion branch.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_relay.py`. The first reproduces the real trace (user prompt has no inspect/verify/create trigger words, so `requires_tool` is False and only the new attempted-action contract can catch the give-up). The second proves ordinary prose answers still pass through untouched.

```python
def conv_plain_task() -> Conversation:
    return Conversation(
        "sys",
        (Turn("user", (TextPart("please make the failing math pass"),)),),
        (ToolDef("Read", "reads", READ_SCHEMA, READ_SCHEMA),),
        GenParams(max_tokens=512, stream=True),
    )


async def test_prose_give_up_after_failed_call_is_fed_back():
    fake = FakeOpenAI()
    fake.push([
        text_chunk("I'll fix it."),
        tool_chunk("c1", "Read", '{"wrong": 1}'),
        finish_chunk("tool_calls"),
    ])
    fake.push([text_chunk("I'll start by using the debugging skill."), finish_chunk("stop")])
    fake.push([tool_chunk("c2", "Read", '{"file_path": "/x"}'), finish_chunk("tool_calls")])
    backend = make(fake)
    metrics: dict = {}
    evs = [e async for e in run(conv_plain_task(), get_profile("qwen"), backend, Settings(), metrics=metrics)]
    assert len(fake.requests) == 3
    assert any(isinstance(e, ToolCall) and e.arguments == {"file_path": "/x"} for e in evs)
    assert metrics["contract_feedback"] == 1
    assert metrics["guard_fires"].get("give_up") == 1
    assert "without a valid tool call" in str(fake.requests[2])
    assert evs[-1].stop_reason == "tool_use"


async def test_pure_prose_answer_passes_through():
    fake = FakeOpenAI()
    fake.push([text_chunk("The answer is 4."), finish_chunk("stop")])
    backend = make(fake)
    metrics: dict = {}
    evs = [e async for e in run(conv_plain_task(), get_profile("qwen"), backend, Settings(), metrics=metrics)]
    assert len(fake.requests) == 1
    assert any(isinstance(e, TextDelta) and "The answer is 4." in e.text for e in evs)
    assert evs[-1].stop_reason == "end_turn"
    assert metrics["contract_feedback"] == 0
```

- [ ] **Step 2: Run tests to verify the right one fails**

Run: `.venv/bin/pytest tests/test_relay.py::test_prose_give_up_after_failed_call_is_fed_back tests/test_relay.py::test_pure_prose_answer_passes_through -q`
Expected: `test_prose_give_up_after_failed_call_is_fed_back` FAILS (only 2 requests made; `contract_feedback` KeyError); `test_pure_prose_answer_passes_through` PASSES (documents existing behavior).

- [ ] **Step 3: Implement the give-up contract**

In `src/harness/relay.py`:

(a) Add the feedback helper after `_append_tool_required_feedback` (near line 152). Wording deliberately avoids verify/create trigger words (see Global Constraints):

```python
def _append_give_up_feedback(conv: Conversation, allowed: list[str]) -> Conversation:
    choices = ", ".join(allowed) or "a valid tool"
    turns = conv.turns + (
        Turn("assistant", (TextPart("[turn ended without a valid tool call]"),)),
        Turn("user", (TextPart(
            "You attempted to act this turn but ended without a valid tool call. "
            "Do not stop and do not answer in free text. Continue the task now by "
            f"calling one of these tools: {choices}."
        ),)),
    )
    return replace(conv, turns=turns)
```

(b) In `run()`, add to the metrics defaults block (after `m.setdefault("first_attempt_constraints", 0)`):

```python
    m.setdefault("contract_feedback", 0)
    m.setdefault("gave_up_honestly", 0)
```

(c) After the line `require_tool_after_invalid_skill = False` (same area as Task 1's totals), add:

```python
    attempted_action = False
```

(d) In the per-attempt state block (where `tool_required_after_invalid_skill = False` is set, near line 313), add:

```python
        give_up_feedback = False
```

(e) In the `elif isinstance(ev, ToolCall):` branch, make the first line record the attempt — change:

```python
            elif isinstance(ev, ToolCall):
                fixed, error = repair_toolcall(ev, conv.tools)
```

to:

```python
            elif isinstance(ev, ToolCall):
                attempted_action = True
                fixed, error = repair_toolcall(ev, conv.tools)
```

(f) In the Done branch's `else:` block (as rewritten in Task 1), insert the give-up check between the `require_tool_after_invalid_skill` check and the final `yield Done(...)`:

```python
                    if (
                        not emitted_valid_call
                        and not buffered_text
                        and ev.stop_reason != "tool_use"
                        and (attempted_action or effective_requires_tool)
                        and attempts < settings.pipeline.repair_retries
                    ):
                        give_up_feedback = True
                        break
```

(g) In the feedback dispatch section after the stream loop, add a new block directly before `if bad_call is None:` (near line 530):

```python
        if give_up_feedback:
            attempts += 1
            m["contract_feedback"] += 1
            increment_guard(m, "give_up")
            suppress_text = True
            conv = _append_give_up_feedback(
                conv, [tool.name for tool in payload_conv.tools]
            )
            continue
```

- [ ] **Step 4: Run the full relay suite**

Run: `.venv/bin/pytest tests/test_relay.py -q`
Expected: all pass. If `test_retries_exhausted_degrades_to_text` fails, the give-up check is firing on `stop_reason == "tool_use"` — recheck condition (f).

- [ ] **Step 5: Commit**

```bash
cd /archive/ai-harness
git add src/harness/relay.py tests/test_relay.py
git commit -m "feat(relay): feed back prose-only give-up turns

A turn where the model attempted a tool call (or the action state requires
one) can no longer end as an empty/prose-only end_turn: it is fed back like
a repair under the repair_retries budget. Closes the 4-second give-up-death
signature from the 2026-07-07 consistency spec. Counters: contract_feedback,
guard give_up.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Honest structured failure when the give-up budget is exhausted

When the model keeps giving up past the retry budget, the client must receive an explicit failure statement — never a bare empty `end_turn` (spec: honest failure is a product success; silence is the defect).

**Files:**
- Modify: `src/harness/relay.py` (failure-text helper; extend the Task 2 give-up check)
- Test: `tests/test_relay.py`

**Interfaces:**
- Consumes: `give_up_feedback` check from Task 2; `total_*` from Task 1; `m["invalid_tool_events"]` (existing).
- Produces: metrics key `gave_up_honestly` set to 1 on this path; a visible `TextDelta` starting with `\n[harness]` before the final `Done`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_relay.py`:

```python
async def test_give_up_exhaustion_ends_with_honest_failure():
    fake = FakeOpenAI()
    fake.push([tool_chunk("c1", "Read", '{"wrong": 1}'), finish_chunk("tool_calls")])
    fake.push([text_chunk("giving up"), finish_chunk("stop")])  # repeats until budget gone
    backend = make(fake)
    metrics: dict = {}
    evs = [e async for e in run(conv_plain_task(), get_profile("qwen"), backend, Settings(), metrics=metrics)]
    # attempt 1: invalid call (retry 1); attempt 2: prose give-up (contract feedback, retry 2);
    # attempt 3: prose again with budget exhausted -> honest failure
    assert len(fake.requests) == 3
    assert metrics["gave_up_honestly"] == 1
    failure_texts = [e.text for e in evs if isinstance(e, TextDelta) and "[harness]" in e.text]
    assert failure_texts and "valid next action" in failure_texts[0]
    done = evs[-1]
    assert done.stop_reason == "end_turn"
    # attempts 2 and 3 reached their usage chunks (10/5 each); attempt 1 broke early
    assert done.input_tokens == 20
    assert done.output_tokens == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_relay.py::test_give_up_exhaustion_ends_with_honest_failure -q`
Expected: FAIL — no `[harness]` TextDelta and `gave_up_honestly == 0` (today the third attempt's suppressed prose vanishes and a bare end_turn is relayed).

- [ ] **Step 3: Implement the honest-failure branch**

In `src/harness/relay.py`:

(a) Add the helper after `_append_give_up_feedback`:

```python
def _honest_failure_text(metrics: dict) -> str:
    events = metrics.get("invalid_tool_events") or []
    detail = ""
    if events:
        last = events[-1]
        detail = f" Last invalid attempt: {last.get('tool')} ({last.get('error')})."
    return (
        "\n[harness] Task step failed: the model could not produce a valid "
        f"next action within the retry budget.{detail} Nothing was applied "
        "silently; this step needs attention.\n"
    )
```

(b) Replace the Task 2 give-up check in the Done branch:

```python
                    if (
                        not emitted_valid_call
                        and not buffered_text
                        and ev.stop_reason != "tool_use"
                        and (attempted_action or effective_requires_tool)
                        and attempts < settings.pipeline.repair_retries
                    ):
                        give_up_feedback = True
                        break
```

with:

```python
                    if (
                        not emitted_valid_call
                        and not buffered_text
                        and ev.stop_reason != "tool_use"
                        and (attempted_action or effective_requires_tool)
                    ):
                        if attempts < settings.pipeline.repair_retries:
                            give_up_feedback = True
                            break
                        m["gave_up_honestly"] = 1
                        yield TextDelta(_honest_failure_text(m))
                        yield Done("end_turn", total_input, total_output, total_cached)
                        return
```

- [ ] **Step 4: Run the full relay suite, then the whole test suite**

Run: `.venv/bin/pytest tests/test_relay.py -q && .venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /archive/ai-harness
git add src/harness/relay.py tests/test_relay.py
git commit -m "feat(relay): honest structured failure when give-up budget is exhausted

Exhausted give-up turns now emit an explicit [harness] failure message and
set gave_up_honestly=1 instead of relaying a bare empty end_turn. Silent
dead sessions become visible, attributable failures.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: End-to-end verification against the live backend (optional if backend is down)

Prove the fix on the exact task that reproduced the death: `fix-test` under the `full` config, several trials.

**Files:**
- None created; results land in `evals/results/` (gitignored or committed per existing convention — do not commit results in this task).

**Interfaces:**
- Consumes: the committed relay changes; live vLLM backend at `http://192.168.0.196:8000/v1`.

- [ ] **Step 1: Check the backend is up**

Run: `curl -s -m 5 http://192.168.0.196:8000/v1/models | head -c 200`
Expected: JSON model listing containing `qwen3.6-27b`. If unreachable, skip Task 4 and report that live verification is pending.

- [ ] **Step 2: Run the failing task family n=5**

Run:
```bash
cd /archive/ai-harness
.venv/bin/python evals/run.py --backend-url http://192.168.0.196:8000/v1 \
  --model qwen3.6-27b --profile qwen --kind vllm \
  --configs full --trials 5 --out evals/results/brick1-verify 2>&1 | tail -20
```
(If `evals/run.py --help` shows a task filter flag, add it to restrict to `fix-test` and `multi-step`; otherwise run all tasks.)
Expected: exit 0, `evals/results/brick1-verify/results.jsonl` written.

- [ ] **Step 3: Check for the death signature**

Run:
```bash
.venv/bin/python - <<'EOF'
import json
rows = [json.loads(l) for l in open("evals/results/brick1-verify/results.jsonl") if l.strip()]
deaths = [r for r in rows if not r.get("input_tokens") and not r.get("output_tokens")]
print(f"trials={len(rows)} zero-token-deaths={len(deaths)} "
      f"successes={sum(bool(r.get('success')) for r in rows)}")
EOF
```
Expected: `zero-token-deaths=0`. Report the success count either way — success-rate improvement is expected but the acceptance criterion for this brick is only the elimination of zero-token deaths.

---

## Self-Review Notes

- Spec coverage: brick 1 of the consistency spec = "contract enforcement + usage fix". Give-up feedback (Task 2), honest failure (Task 3), usage accumulation (Task 1) — covered. The spec's done-claim gate already exists and blocks (verified by existing `test_done_claim_after_edit_requires_verification`); no task needed. Per-session contract cap and remaining bricks (ruler v2, first-attempt constraints) are later bricks by design.
- Type consistency: `total_input`/`total_output`/`total_cached` defined in Task 1, consumed in Tasks 2–3; `give_up_feedback` defined in Task 2, extended in Task 3; helper names match between definition and dispatch.
- The give-up condition includes `not buffered_text` so it can never double-fire with the buffered-text path, and `stop_reason != "tool_use"` so `test_retries_exhausted_degrades_to_text` is unaffected.

## Execution Deviation (recorded 2026-07-07)

Task 2 as planned used `attempted_action` (any ToolCall event seen this request)
in the give-up condition. That broke the two cross-turn loop-break tests: loop
feedback explicitly invites a free-text conclusion ("state your conclusion"),
so a prose-only end_turn after it is a legitimate answer, not a give-up. The
implemented condition instead uses `expects_tool_retry`, set True only by
feedback paths that explicitly demand a tool retry (invalid tool call,
action-state block, invalid-skill retry, give-up feedback itself), cleared by
loop-break feedback and by any emitted valid call. `effective_requires_tool`
was also dropped from the give-up check — the buffered-text path already
enforces it on unsuppressed attempts, and suppressed post-loop-break prose must
stay accepted (encoded in test_cross_turn_loop_ignores_bash_description_metadata).
