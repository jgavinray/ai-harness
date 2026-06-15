# Runtime Control Plane for Local Model Reliability

## Goal

Make `ai-harness` a stricter model-edge runtime for local Qwen-class models:
enforce legal next actions, compact context before exhaustion, prevent repeated
deterministic mistakes, and reserve expensive reasoning models for semantic
review instead of operational cleanup.

This plan intentionally does not duplicate `agentic-os` orchestration,
trajectory storage, memory, or request-classification work. `agentic-os` owns
long-lived memory and policy learning. `ai-harness` owns protocol shaping,
tool-call enforcement, context budgeting, and backend-facing reliability.

## Current Failure Modes

- The critic spends expensive qwen80 tokens on deterministic issues such as
  path aliases, bad grep syntax, missing parent directories, and tool misuse.
- The worker model often attempts to bypass tool schemas or use Bash where a
  structured tool would be safer.
- Context compaction is reactive. It triggers only after the request exceeds the
  computed budget instead of proactively compacting near a safe threshold.
- Tool pruning exposes a small schema set and a catalog. Weaker models can call
  catalogued tools by name without seeing enough schema detail up front.
- The advertised context window is treated as the practical context window, even
  though model quality degrades before the hard limit.

## Design Principles

1. Deterministic failures must be handled by software, not qwen80.
2. The worker should see only the next legal action surface whenever possible.
3. Tool-call constraints should apply before the first bad call, not only after
   a repair retry.
4. Context should compact before quality falls off, not only before hard
   overflow.
5. Logs must capture fine-tuning substrate: bad call, correction, corrected
   call, and outcome.
6. Runtime metrics should make skipped critic calls, guard rewrites, and context
   compaction visible without external report scripts.

## Phase 1: Instrument Before Changing Policy

Add request-level metrics without changing behavior.

New fields:

- `context_tokens_before`
- `context_tokens_after`
- `context_budget`
- `context_compacted`
- `compaction_reason`
- `turns_elided`
- `tool_results_truncated`
- `action_state`
- `allowed_tools`
- `preflight_decision`
- `preflight_reason`
- `critic_eligible`
- `critic_skipped_reason`

Dashboard additions:

- invalid tool rate
- preflight rewrites
- preflight denies
- critic skipped
- context compactions
- context before/after compaction

Validation:

- Unit tests for emitted metrics on normal, compacted, and repaired requests.
- `/stats` exposes rolling counters for all new metric families.

## Phase 2: Deterministic Tool Preflight

Create a preflight layer that runs after model output is parsed and repaired,
but before the tool call is emitted to the client.

Decision types:

- `allow`: emit tool call unchanged
- `rewrite`: mutate arguments and emit corrected call
- `deny`: append deterministic feedback and retry, without calling qwen80

Initial checks:

- known workspace path alias rewrites
- paths outside allowed roots
- nonexistent absolute paths
- missing parent directory before `Write` or shell redirection
- `grep "a|b"` without `-E`
- Bash command that should be `Read`, `Grep`, `Glob`, or `Edit`
- repeated identical failing command

Metrics:

- `preflight_rewrites`
- `preflight_denies`
- `preflight_reasons`
- `original_arguments`
- `rewritten_arguments`
- `tool_success_after_preflight`

Validation:

- Bad path aliases are rewritten without critic involvement.
- Missing parent directory produces deterministic feedback.
- Bad grep syntax is denied or rewritten.
- Existing workflow guards continue to fire in the same order.

## Phase 3: Runtime Action State Machine

Add `src/harness/action_state.py`.

The state machine determines legal next actions from conversation history,
recent tool results, plan status, and edit/verify state.

States:

- `inspect`: only `Read`, `Grep`, `Glob`, `LS`
- `edit_existing`: `Edit` or `MultiEdit`, only after the target file was read
- `create_file`: `Write` or mkdir-first Bash, under allowed roots only
- `verify`: verification/build/test Bash only
- `finish`: text completion allowed
- `blocked`: deterministic feedback required before another backend call

Relay behavior:

- Shape surfaced tools from state.
- If a tool is required, suppress free-text-only completions.
- If a state is blocked, skip backend call and append deterministic feedback.

Validation:

- Edit is unavailable before read.
- Verify state does not permit additional edits unless verification fails.
- Finish state rejects completion after unverified edits.
- Tool menu changes are logged and visible in `/stats`.

## Phase 4: First-Attempt Constrained Tool Calls

Today constraints are mostly a repair mechanism. Move constraints earlier.

Behavior:

- If the action state requires one tool, apply that tool schema on attempt zero.
- If a small allowed set exists, use a union schema where backend support allows.
- If backend does not support union constraints, use deterministic feedback to
  require a specific next tool.
- Continue using retry constraints for malformed second attempts.

Validation:

- vLLM receives guided JSON on the first required-tool attempt.
- Invalid tool-call rate drops in dashboard metrics.
- Free text is not emitted when a tool is required.

## Phase 5: Bash Command Classes

Classify Bash commands before emission.

Classes:

- `inspect`
- `create_dir`
- `verify`
- `build`
- `test`
- `dangerous`
- `unknown`

Policy:

- Prefer structured tools for inspect/edit operations.
- Allow `create_dir` when it satisfies a missing parent directory preflight.
- Allow `verify`, `build`, and `test` after edits or when requested.
- Deny `dangerous` unless explicitly requested by the user.
- Escalate repeated `unknown` commands only after deterministic retries fail.

Validation:

- Shell redirects to missing dirs are blocked or converted to mkdir-first flow.
- Build/test commands are allowed in verify state.
- Read-like shell commands are redirected toward structured tools.

## Phase 6: Proactive Context Compaction

Extend `HistoryStage` with effective context thresholds.

New config:

```toml
[pipeline]
effective_context_window = 131072
compact_at_ratio = 0.80
compact_target_ratio = 0.50
```

Behavior:

- Count context before backend dispatch.
- Compact when `context_tokens >= effective_context_window * compact_at_ratio`.
- Compact down to `effective_context_window * compact_target_ratio`.
- Preserve the opening task, active plan/status, recent tool results, current
  file facts, and unresolved deterministic feedback.

Validation:

- Compaction occurs before hard budget exhaustion.
- Recent turn protection remains intact.
- KV prefix invalidation is reduced relative to reactive compaction.
- Metrics show before/after token counts and reason.

## Phase 7: Critic Eligibility Gate

Add deterministic gating before calling qwen80 critic.

Skip critic when:

- path alias rewrite occurred
- schema failure can be retried
- missing parent directory is known
- edit-before-read guard fired
- grep syntax can be corrected
- repeated exact command is detected

Allow critic when:

- build/test failure has multiple plausible semantic causes
- risky C/API/kernel/interface change is detected
- plan conflict is semantic rather than mechanical
- deterministic guard repeats and fails to correct behavior

Metrics:

- `critic_eligible`
- `critic_skipped_reason`
- `critic_saved_turn_estimate`
- `critic_repeated_feedback_hashes`

Validation:

- Known deterministic errors do not call qwen80.
- Semantic build failures still call qwen80.
- Recent critic revise rate falls while task success remains stable or improves.

## Phase 8: Fine-Tuning Data Capture

Capture protocol-level training examples.

For each bad-to-good sequence:

- request/session identifiers
- action state
- allowed tools
- bad tool call
- preflight decision
- deterministic feedback or critic feedback
- corrected tool call
- tool result outcome
- next-turn success/failure

Important: `critic_feedback` must be stored in full, without truncation, because
it is fine-tuning data.

Validation:

- JSONL records contain complete critic feedback.
- Repeated feedback hashes can be grouped.
- A corpus builder can reconstruct bad call -> correction -> outcome examples
  from logs without traces.

## Success Criteria

- qwen80 critic calls drop for deterministic failures.
- Invalid tool-call retry rate falls.
- Repeated path/tool mistakes fall sharply.
- Context compaction occurs before quality collapse.
- Average qwen27 TTFT decreases due to fewer full-context re-prefills.
- Dashboard shows whether improvements are from deterministic controls or
  critic reasoning.
- Fine-tuning corpus contains complete feedback and corrected action pairs.

## Non-Goals

- Do not rebuild `agentic-os` memory, trajectories, context packs, or policy
  classification inside `ai-harness`.
- Do not expose model chain-of-thought.
- Do not make qwen80 the default planner for deterministic workflow errors.
- Do not rely on prompt wording when a software guard can enforce the invariant.

