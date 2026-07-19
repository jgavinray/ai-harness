# Adversarial review loop ("the auditor"), 2026-07-19

Extends the consistency design (2026-07-07, escalation ladder) and obeys the
default-open enforcement amendment (2026-07-11). Approved intent from the
owner: responses are still reaching the client that are "not consistent with
reality"; before a response returns, a hostile adversarial agent and the
proposing agent must **come to consensus** that it is proper, accurate
(evidenced), and within the original ask. Tokens are free on local
hardware; the loop is also a **sensor** — its logged catches are the data
from which future deterministic guard rules are mined.

## Problem

Four observed failure shapes, all confirmed live by the owner:

1. **Unsupported final claims** — answers assert things ("the bug is in X",
   "tests pass") the transcript's tool results do not support.
2. **Confabulated code/APIs** — descriptions of files, functions, or APIs
   that do not exist.
3. **Scope drift** — internally accurate responses that answer something
   other than the original ask.
4. **Wrong mid-task actions** — bad tool calls/edits during the task, before
   any final prose.

The existing deterministic layer (guards, action states, done-claim
verification) binds on mechanical facts and cannot see semantic
groundedness. We have not yet been collecting the data needed to write
deterministic rules against these shapes — the review loop collects it.

## Constraints that shape the design

- **Remote proxy (hard).** The harness runs on hyper01; source code lives on
  the client host and tools execute in the client's Claude Code. The
  reviewer's only admissible evidence is the conversation transcript (which
  contains every Read/Grep/Bash result the executor saw) plus the original
  user request. "You assert X but never read the file" is a valid, actionable
  objection.
- **One live backend.** qwen27 (vLLM, 192.168.0.196) serves executor and
  reviewer. Reviewer roles on a stronger backend (e.g. the 80B-thinking box)
  slot in later via config only.
- **Law 2 prefix stability.** Nothing from the debate may enter the
  executor's persistent conversation. Reviewer/counter-review prompts reuse
  the executor's byte-identical conversation prefix so the backend's prefix
  cache eats the prefill.
- **Client timeouts (physical).** The Claude Code CLI aborts requests that
  stall past its stream/request timeouts. An aborted request is strictly
  worse UX than an imperfect answer.
- **Law 1.** `enforce` mode ships only behind a measured eval delta.
- **Law 6 / default-open.** Verdicts subtract (objections name specific
  unsupported claims); the reviewer never enumerates what a response is
  permitted to contain.

## Design

### Scope

**Every turn** — tool-call turns and prose/answer turns alike, including
subagent sessions. Maximum sensor coverage was chosen deliberately over
cheaper trigger subsets.

### Debate protocol (subagent-shaped, fresh context per round)

Three parties; each is a separate backend request with its own context
window. The client-visible session never contains debate text.

1. **Proposal** — the executor's candidate turn, produced by the normal
   relay path (unchanged).
2. **Hostile reviewer** — fresh context: session transcript (byte-identical
   to the executor's prefix → cached prefill) + candidate + adversarial
   instruction: *every claim and every tool-call argument must be evidenced
   in the transcript or the original ask; you cannot read files; unverified
   is not acceptable; also check the candidate against the original request
   for scope drift.* Output verdict: `approve`, `objection` (numbered,
   evidence-citing), or `flag` (log-worthy, not block-worthy).
3. **Counter-review** — fresh context: transcript prefix + candidate +
   objection. The proposal side either **concedes** (revised candidate,
   generated through the existing relay retry path with the objection as
   feedback) or **rebuts** with transcript citations.
4. Reviewer round N+1 sees transcript prefix + current candidate + only the
   **latest** objection/rebuttal pair — never the accumulated debate log.
   Per-round context is constant regardless of debate length.

**Consensus** = the reviewer approves the current candidate or accepts the
rebuttal. Then the candidate ships.

### Termination: consensus, not a round cap

There is **no `max_rounds`**. Two valves fire only on pathology:

1. **No-progress detection.** A round whose critique is substantively
   identical to a prior round's (normalized-hash comparison, same idiom as
   `repair/degenerate.py`), or whose candidate is unchanged from the prior
   round, is livelock — two instances of the same model disagreeing
   forever. Ship the current best candidate; log the unresolved objection.
2. **Client lifeline.** SSE keepalive pings stream during the debate. When
   wall-clock approaches the configured client-timeout horizon
   (`review.client_deadline_s`), ship the current best candidate rather
   than letting the CLI abort.

On either valve, done-claim turns get a `[harness] reviewer objection: …`
note appended; other turns ship untouched with the objection logged. A
debate that keeps making progress may run as long as the client will wait.

### Failure isolation

Reviewer inference error or timeout → the executor's response ships
untouched and a `reviewer_error` record is logged. The reviewer can never
take down serving (same contract as flywheel jobs).

### Modes

`review.mode = "off" | "shadow" | "enforce"`.

- **shadow**: the full debate runs and logs, but the original candidate
  always ships unmodified. Zero behavioral risk; data flows from day one.
- **enforce**: revised/consensus candidates ship. Gated by Law 1 (below).

### Data plane (the rule-mining sensor)

Every round appends one record to `logs/reviews/YYYY-MM-DD.jsonl`:
session key, request id, turn kind, round number, candidate hash, verdict,
objection/rebuttal text, claims-checked list, outcome
(consensus/deadlock/deadline/reviewer_error), tokens and wall-clock spent.
DuckDB indexes it as a derived table. A nightly flywheel job aggregates
recurring objection patterns into `logs/review_patterns/<date>.json` and a
morning-readable report — the graduation pipeline from adversarial catches
to deterministic guard rules. JSONL remains the source of truth (Law 5).

### Config

`[review]` (extending the existing section): `mode`, `debate_roles`,
`client_deadline_s`, `keepalive_interval_s`, `reviews_dir`. The debate has
**no token limits of its own** (owner decision 2026-07-19): each reviewer /
counter-review inference inherits the `max_tokens` the client requested for
the turn. Reviewer backend selection stays role-based (`review` role on
qwen27 now; movable by config alone).

### Rollout (Law 1)

1. Build both modes TDD-first (Law 3).
2. Turn on **shadow** for live traffic immediately.
3. Add a **groundedness eval family**: tasks with planted bait where the
   correct answer requires reading a file, and a checker that scores
   whether shipped claims match repo reality.
4. Certify **enforce**: full envelope suite (8 families, n=20) + the new
   family, compared against the current 171/171 baseline. The run also
   yields the first empirical consensus-latency distribution.
5. Flip `enforce` only on a clean delta; the toml records provenance as
   usual.

## Testing

- Unit: verdict parsing (approve/objection/flag, malformed reviewer output),
  round loop (concede path, rebut path, consensus), no-progress hash valve,
  client-deadline valve, reviewer-failure isolation, byte-stability of the
  reviewer prompt prefix against the executor's rendered prefix, shadow mode
  never mutates the shipped response, review records written per round.
- Eval: groundedness family (new) + full envelope regression.

## Non-goals

- No server-side file reading by the reviewer (impossible; remote proxy).
- No accumulated multi-round debate transcript (context stays constant).
- No enumerated allowlist of acceptable response content (Law 6).
- No third always-on service: the reviewer is in-process in the serving
  container; only the nightly pattern job runs in the flywheel (Law 5).
