# Strong Reasoning Model Routing SOW

## Goal

Introduce a stronger reasoning model as a first-class runtime capability in the
harness. The stronger model should be used when the user is asking for thinking:
explanation, research, architecture, comparison, diagnosis, or conceptual
analysis. It should also support the executor as an automatic planner and critic
for risky coding moments.

The user should not need to say `plan`, `reasoning`, or any other magic prefix
for normal use. Routing should be automatic by request intent, with explicit
overrides left as a future convenience rather than the core design.

## Current State

- The router already supports role-based backend selection with `main`,
  `subagent`, `fast`, `candidate`, and capability matching such as `vision`.
- Planning already exists as a sidecar path: `[planning]` calls a `plan` role
  backend once per main session and injects a compact `## Execution plan`.
- Workflow guards already catch deterministic failure modes such as editing
  without reading, claiming done before verification, plan drift, repeated tool
  loops, and invalid tool-call repair retries.
- The improvement loop already exists: traces, evals, candidate backends,
  promotion gates, scaffold relaxation, corpus generation, and LoRA scaffolding.

## Scope

### Direct Reasoning Route

Add a live `reasoning` role for stronger-model interaction.

Use this route when the latest user turn is primarily asking the system to:

- explain code or behavior;
- analyze a problem;
- compare approaches;
- reason through architecture;
- review a design;
- summarize or research information;
- diagnose without immediately editing files;
- answer conceptual questions.

Do not use this route when the user is asking to:

- implement a feature;
- fix a bug;
- edit, create, or delete files;
- run a command as the main task;
- complete an agentic coding workflow.

Those remain `main` or `subagent` executor traffic.

### Per-Turn Classification

Classify each request independently. A session can move from reasoning to coding
and back again without permanently changing session ownership.

Store router affinity by `(session_key, role)` instead of only `session_key`, so
a reasoning turn does not overwrite the executor model's warm KV affinity.

### Read-Only Reasoning Tools

When traffic is routed to `reasoning`, expose only read-only/inspection tools.
The intended allowlist is:

- `Read`
- `Grep`
- `Glob`
- `LS`
- `WebFetch`
- other explicitly read-only tools already present in a Claude Code request

Block or repair-feedback attempts to use mutating tools such as:

- `Edit`
- `MultiEdit`
- `Write`
- mutating `Bash` commands

The reasoning model can inspect context to give grounded answers, but it should
not perform code changes. Implementation work remains the executor model's job.

### Planner

Keep planning automatic. When `[planning].enabled = true` and request role is
`main`, call the `plan` role backend once per session. The user does not need to
ask for a plan.

The planner output remains advisory and compact:

- implementation-grade steps;
- relevant risks;
- verification expectations;
- no hidden chain-of-thought;
- no authority to override harness/system rules.

### Runtime Review / Critic

Add a separate `review` role for the stronger model acting as a critic of risky
executor behavior.

The reviewer should not watch every token or replace the executor. It should run
only at harness-observed checkpoints:

- executor claims completion after edits without verification;
- executor drifts from the current plan;
- executor repeats the same tool call;
- executor repeatedly emits invalid tool calls;
- executor encounters test failure output and appears to continue blindly.

The review response should be short and structured: approve, revise, or no-op,
with one concise feedback message if revision is needed. The relay should append
that feedback through the existing retry path.

If review fails or no review backend is configured, the harness should continue
with existing deterministic guards.

### Improvement Loop Preservation

The stronger model's reinforcement learning is treated as a property of the
model itself. The harness does not perform online RL.

The existing improvement loop remains the source of operational truth:

- request logs;
- traces;
- eval outcomes;
- candidate backend shadow evals;
- promotion gates;
- scaffold relaxation;
- corpus generation;
- future fine-tuning jobs.

The strong reasoning route should add metrics and traces that make the loop more
useful, not replace it.

## Configuration

Example backend roles:

```toml
[[backends]]
name = "strong-reasoner"
kind = "vllm"
base_url = "http://reasoner:8000/v1"
model = "strong-reasoning-model"
profile = "qwen"
context_window = 131072
roles = ["reasoning", "plan", "review"]

[[backends]]
name = "executor"
kind = "vllm"
base_url = "http://executor:8000/v1"
model = "coding-executor-model"
profile = "qwen"
context_window = 131072
roles = ["main", "subagent"]

[[backends]]
name = "fast"
kind = "llamacpp"
base_url = "http://fast:8080/v1"
model = "small-fast-model"
profile = "qwen"
context_window = 32768
roles = ["fast"]
```

New config sections should be minimal:

```toml
[routing]
reasoning = true
reasoning_readonly_tools = ["Read", "Grep", "Glob", "LS", "WebFetch"]

[review]
enabled = false
max_chars = 6000
max_tokens = 500
triggers = ["plan_drift", "verify_after_edit", "loop_break", "invalid_tool_retry"]
```

## Metrics

Add request log fields where applicable:

- `role = "reasoning"` for direct strong-model turns;
- `routing_intent`;
- `routing_reason`;
- `review_generated`;
- `review_trigger`;
- `review_action`;
- `review_error`;
- existing planning, guard, repair, trace, and eval metrics unchanged.

## Acceptance Criteria

- Explanation and architecture questions route to `reasoning` automatically.
- Implementation and bug-fix requests still route to `main`.
- Reasoning turns can use read-only tools but cannot mutate files.
- Reasoning affinity does not steal or overwrite main executor affinity.
- Planning remains automatic and does not require user prefixes.
- Runtime review can critique risky executor checkpoints without taking over
  execution.
- Review failure degrades to existing deterministic guard behavior.
- Existing improvement-loop scripts and eval-gated promotion flow continue to
  work.
- Tests cover routing classification, role-specific affinity, read-only tool
  filtering, review triggers, and no-regression behavior for existing router and
  planning tests.

## Out Of Scope

- Online reinforcement learning inside the harness.
- Letting the reasoning model perform unrestricted code edits.
- Replacing the executor model for all coding turns.
- Requiring users to use explicit prefixes for normal reasoning behavior.
- Promoting any model without eval evidence.
