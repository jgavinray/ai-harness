# Brick 1 verification — turn contract + action-state trap fix

Backend: vLLM qwen3.6-27b (Qwen3.6-27B-int4-AutoRound) at 192.168.0.196:8000.
Config: `full`, tasks fix-test + multi-step, 5 trials each, via the real
Claude Code CLI. Raw rows: `evals/results/brick1-verify/` (before the
action-state fix) and `evals/results/brick1-verify2/` (after).

| run | task | success | zero-token deaths | timeouts | avg wall |
| --- | --- | --- | --- | --- | --- |
| before (33f5f2e) | fix-test | 1/5 | 0 | 2 | 130 s |
| before (33f5f2e) | multi-step | 0/5 | 1 | 1 | 89 s |
| after (0edd4ab, action-state fix) | fix-test | 5/5 | 0 | 0 | 15 s |
| after | multi-step | 4/5 | 0 | 1 | 79 s |

Notes:

- The single remaining failure (multi-step trial 4) **passed its check**
  ("all tests passed") but the session ran until the 300 s timeout — the
  runaway-turn class (see below), not a task failure. Counting fixed tasks,
  the after-run is 10/10.
- Zero-token dead sessions: eliminated (brick 1 acceptance criterion).
- `contract_feedback` and `gave_up_honestly` stayed 0 in the after-run: with
  the action-state trap removed, the model never needed rescuing. The
  contract is a backstop, and the eval proves it doesn't fire spuriously.

## What the before-run exposed (root cause of the June→July regression)

The action-state sprint (commits after 2026-06-12) introduced two traps,
found because both eval prompts contain the words "verify"/"create new
files":

1. Free-text intent detection locked entire sessions into `verify` (Bash +
   read-only) or `create_file` state from the task brief alone; the model
   correctly diagnosed each task and was denied every Edit/Write attempt.
   Trial traces end with
   `[action state denied Write: expected Edit, MultiEdit, Read, Grep, ...]`.
2. No reachable state allowed `Write` at all, so creating a new file
   (multi-step's whole point) was impossible.

Fix (0edd4ab): free-text intent binds only for short
imperative instructions (≤64 chars, e.g. "run tests"); `Write` joined the
inspect and edit_existing states. The unverified-edit→verify contract,
plan-step verify, and edit-state Bash discipline are unchanged (all
encoded in tests that still pass).

## Known remaining failure class (next brick)

**Runaway turns:** one before-run trial hung for 300 s producing only
thinking tokens (1 request, 0 output, no reasoning ceiling in eval
configs); one after-run trial finished the task then failed to end the
session before timeout. Both are "the model doesn't stop" — a relay-level
turn/reasoning ceiling plus a session-level settle guard, to be designed
under the consistency spec's contract table (docs/superpowers/specs/
2026-07-07-daily-driver-consistency-design.md).
