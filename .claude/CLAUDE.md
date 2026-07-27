# Operating rules (global)

## Task protocol — follow in order, every task
1. If `.claude/wf-state.json` exists, resume the workflow at its recorded state and consume its `feedback` field first.
2. Else if `TASKS.md` exists, work only the first unchecked item via the workflow state machine (PLAN → IMPLEMENT → TEST → VERIFY → REVIEW → DOCUMENT → COMMIT; see the workflow skill). If neither exists, treat the user prompt as the task and start at PLAN.
3. On any gate failure, return to the fail-state defined by the workflow table carrying the blocker's report verbatim; address the FIRST item, re-run the failed gate directly.
4. Run `.claude/verify.sh` before ever claiming completion. Do not stop or summarize while it fails.
5. On pass: check the item off in `TASKS.md` with a one-line note; write the transition to `.claude/wf-state.json` before acting on it.

## Hard rules
- Never state that work is complete without having run verification in this turn.
- Re-read a file immediately before editing it. Never edit from memory.
- One logical change per edit. Small diffs.
- If the same command fails twice with the same error, change approach — do not run it a third time unchanged.
- Persist state to disk (`TASKS.md`, `NOTES.md`), not to conversation memory. Assume context will be lost.
- No placeholders, no stub functions, no "TODO: implement", no partial snippets. Complete, runnable deliverables only.
- Deterministic artifacts: sorted keys/lists, pinned versions, no timestamps or random values in generated output.
- Use tools to find answers. Do not ask the user for information a command can produce.
- When the user corrects your behavior ("stop doing X", "always Y"), comply AND append the correction verbatim with context to `.claude/proposals/inbox.md` — corrections must outlive this session.
- When blocked by a genuine decision (destructive action, ambiguous requirement with divergent outcomes), stop and ask one specific question. Otherwise do not ask.

## Output discipline
- No preamble, no recap of these rules, no apologies.
- Final message per task: what changed (files), verification result, next unchecked task if any.
