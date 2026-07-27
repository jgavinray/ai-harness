---
description: Drive TASKS.md to completion, one item at a time, gated by .claude/verify.sh
---
# task-loop

Purpose: convert a goal into a durable on-disk checklist, then burn it down deterministically. The Stop hook enforces the gate; this skill defines the procedure.

## Setup (only if TASKS.md is missing)
1. Decompose the goal into 3–15 independently verifiable items. Each item must state its acceptance check.
2. Write `TASKS.md`:

```markdown
# TASKS
- [ ] T1: <action> — verify: <command or observable condition>
- [ ] T2: ...
```

3. Ensure `.claude/verify.sh` exists, is executable, and actually exercises the acceptance checks. If it doesn't cover a task's check, extend it first — the verifier is the definition of done.

## Loop (repeat until no unchecked items)
1. Read `TASKS.md`. Select the FIRST unchecked item only. Ignore all others.
2. Implement the smallest change satisfying that item's check.
3. Run `.claude/verify.sh`. On failure: fix the first failure, re-run. Never proceed past a red verifier.
4. On pass: edit `TASKS.md` — mark the item `[x]` and append `(done: <files touched>)`.
5. Commit if the project uses git: `git add -A && git commit -m "T<n>: <item summary>"`.
6. Return to step 1.

## Rules
- State lives in TASKS.md, never in memory. After any interruption, re-read it and resume at step 1.
- One item per iteration. Do not batch.
- If an item fails 3 consecutive verify attempts, append a `BLOCKED:` line under it with the exact error, skip to the next item, and report all BLOCKED items at the end.
- Final message when the list is complete: item count done/blocked, verifier status, commit list.
