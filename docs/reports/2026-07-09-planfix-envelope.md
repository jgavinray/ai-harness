# Envelope re-certification after the plan-step lockout fix

Runs: `evals/results/envelope-27b-planfix/` + `envelope-27b-planfix2/`
(the first run was interrupted at 111/111 during multi-step; the second
completed the remaining families — multi-step therefore has 31 trials).
Code: commits 5d7d03b (plan status line no longer binds enforcement +
code-review family) and dc41481 (`cd X && cmd` classification). Backend:
qwen3.6-27b (int4-AutoRound) on .196, `full` config, real Claude Code CLI.

## Why this run exists

The 2026-07-08 certification (140/140) was followed by the first real
workload — a read-only code review — being **completely unusable**: 55/58
requests locked in verify state by `plan_verify_step`, 48 preflight
denies including `git diff`, `git status`, `ls`, the project's linter,
and `echo`. The plan status line advances by tool-call count, so any
session longer than its plan pins to the final "Verify …" step; four
gates keyed enforcement on that fiction. The synthetic envelope never
caught it because every eval session was shorter than its plan.

## Verdicts (all supported)

| family | success |
| --- | --- |
| add-endpoint | 20/20 |
| code-review (new) | 20/20 |
| find-and-report | 20/20 |
| fix-test | 20/20 |
| long-horizon | 20/20 |
| multi-step | 31/31 |
| rename-refactor | 20/20 |
| tool-discovery | 20/20 |

**171/171.** The `code-review` family reproduces the failing workload
shape (long read-only review over uncommitted changes, prose deliverable,
prompt full of verify/check/lint trigger words, runner `setup.sh` overlay)
and was 100% deadlocked on the pre-fix code (smoke evidence:
`evals/results/code-review-smoke`, 3/3 after the fix, 0 denies).

## Standing lesson

Fourth occurrence of the word-trigger bug shape, first from
model-generated text (plan steps) rather than user prompts. Rule
reaffirmed as code: plan state is informational only; enforcement keys
exclusively on mechanical facts (unverified edit counts, short imperative
instructions). And: the envelope only certifies workloads it contains —
real-workload families (flywheel phase 3) are the defense, not more
synthetic ones.
