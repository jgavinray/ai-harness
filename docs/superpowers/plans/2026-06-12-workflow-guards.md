# Workflow Guards Implementation Plan

**Status:** done 2026-06-12.

**Goal:** Extend the relay's existing degenerate-output and repeated-call
protection into deterministic workflow guards with per-guard counters.

## Tasks

- [x] Add `[pipeline]` toggles for workflow guards:
  `workflow_guards`, `guard_edit_without_read`, and
  `guard_verify_after_edit`.
- [x] Add read-before-edit guard: an `Edit`/`MultiEdit` call for a file not read
  in surviving history is converted into feedback and retried under the existing
  relay retry budget.
- [x] Add verify-before-done guard: after an `Edit`/`Write`, a completion claim is
  buffered and converted into feedback unless a relevant `Bash` verification
  command has run since the edit.
- [x] Report counters as `guard_fires.<name>` in relay metrics, including the
  existing repeated-call loop breaker as `same_approach`.
- [x] Add relay tests for guard firing, retry feedback, disabled guards, and
  counter recording.
- [x] Document the guard toggles in `harness.toml.example`.

## Verification

Run:

```bash
.venv/bin/python -m pytest tests/test_relay.py -q
.venv/bin/python -m pytest tests/ -q
```

Observed:

```text
14 passed
166 passed
```
