# claude-harness

Rebuilt Claude Code / opencode driving config: rules, hooks, agents, one skill, project templates. Designed for non-frontier local models (Qwen-class via an Anthropic-compatible endpoint): short imperative rules, on-disk state, and a hard verification gate instead of trusting the model's self-report.

## Core mechanism: the Stop gate
`hooks/stop-gate.sh` runs on every `Stop` event. If the project has an executable `.claude/verify.sh` and it exits nonzero, the hook emits `{"decision":"block","reason":<failure tail>}`, which prevents the model from stopping and injects the failure output as feedback. The model iterates until `verify.sh` exits 0 or the cap (`CLAUDE_STOP_GATE_MAX`, default 8) is hit. "Done" is defined by the verifier, not by the model — that's what makes the result deterministic.

## Outer loop: `wf-run.sh` (context management + task graph)
The Stop gate handles the tight in-session loop; `wf-run.sh` is the outer driver that runs the whole graph to completion with bounded context:

- **Graph**: `.claude/tasks.json` — nodes `{id, desc, verify, deps[], status, attempts}` emitted by the planner. Driver dispatches the first READY node (todo + all deps done); blocked nodes make dependents unreachable by design.
- **Fresh context per iteration**: each pass spawns a new headless session (`claude -p` / `opencode run` / `--backend cmd` for custom runners) seeded with only the node, `wf-state.json`, and the newest feedback record. Disk is memory; conversation history is disposable — this is the context-management strategy, not an add-on to it.
- **Driver-side gates**: after each session, wf-run itself runs the node's `verify` command AND `.claude/verify.sh`; the agent's self-report is never trusted. Failure → `wf-run` record appended to `feedback.jsonl`, node returns to READY with feedback waiting; `WF_NODE_ATTEMPTS` (default 3) failures → blocked. `WF_MAX_ITER` caps the run.
- Exit codes: 0 all done, 73 finished with blocked nodes, 74 iteration cap.

- **State restoration**: every attempt starts from a snapshot commit; on gate failure the driver `git reset --hard` + `clean -fd -e .claude` and restores state files, so attempt N+1 gets a clean tree plus feedback — never the previous attempt's half-broken edits. Passes are committed (`wf-run: <id> done`).
- **Crash recovery**: startup sweeps `doing`→`todo`; every session runs under `WF_SESSION_TIMEOUT` (default 1800s), timeouts recorded as feedback ("decompose the task").
- **Flake detection** (`WF_FLAKE_CHECK=1`): a passing gate is immediately re-run; pass-then-fail is reported as FLAKY VERIFICATION and treated as failure — protects the definition of done itself.
- **Transcript corpus**: every session's output lands in `.claude/transcripts/<id>-attempt<N>.log` with a sidecar label `{task, attempt, outcome: pass|fail|timeout, backend, model, gate_exit}` — both pass and fail trajectories kept, distillation-ready. `WF_CLAUDE_ARGS`/`WF_OPENCODE_ARGS` pass through verbatim for richer output formats.
- **Model routing**: `WF_MODEL`/`--model` sets the default; a per-node `"model"` field overrides it — route planning/review to a strong model, implementation to local.
- **Ownership**: driver owns `tasks.json` (flock-serialized when available), agents own `wf-state.json`/`NOTES.md`, `feedback.jsonl` is append-only for everyone.

The workflow core also gained a **Small-model discipline** section (rendered into the skill and orchestrator): anchor task-id every turn, one action per turn, quote feedback verbatim, re-read after every edit, targeted reads only, invalid report format = regenerate, never negotiate with a gate, change strategy not intensity, say less. Each rule maps to an observed sub-frontier failure mode.

## Self-improvement loop
Four nested loops, fastest to slowest. L0 (per attempt): gate fail → feedback → rollback → fresh retry. L1 (per run): `wf-stats.sh` aggregates transcripts + feedback into health signals — the key one is **feedback efficacy**, the fail→pass rate on the attempt following feedback, which tells you whether the loop is learning or re-rolling dice. L2 (harness): the `retro` agent turns >=2-occurrence patterns into evidence-backed proposals in `.claude/proposals/` (target, metric, exact diff, risk); **you** flip Status to approved/rejected; approved proposals run as normal gated wf-run nodes; the next retro measures whether the metric moved and proposes REVERT if not. Rejections are permanent memory — retro never re-proposes them, so your judgment compounds into the system. Two safety properties: the **gate immutability rule** (no change touches gate logic and gate-judged behavior together) and **correction capture** (mid-session "stop doing X" corrections are appended to `.claude/proposals/inbox.md` instead of dying with the context — CLAUDE.md rule). L3 (model, optional): transcripts accumulate as labeled corpus for free; teachers exist via per-node model routing (Opus trajectories), rejection sampling (gate-passed local trajectories), or big-local→small-local.

**Corpus tooling**: `wf-corpus.sh` converts labeled transcripts into chat-format SFT JSONL (axolotl/LLaMA-Factory/torchtune compatible): `corpus-pass.jsonl` (gate-passed trajectories — rejection-sampled positives, verifier as teacher) and `corpus-repair.jsonl` (passes whose prompt carried prior-failure feedback — fail→pass repair pairs, the feedback-following signal). Prompts are persisted per attempt by wf-run. License hygiene is structural, not optional: records with backend `claude` or model matching `claude|anthropic` are excluded from both sets with no override flag — those transcripts serve stats/retro only. `WF_CORPUS_EXCLUDE_MODELS` extends the filter for other restricted providers.

**The harness gates itself**: `./verify.sh` at the repo root runs `tests/harness-test.sh` — 34 checks covering script syntax, JSON validity, plugin import, render-drift (core sections present in both renderings), stop-gate block/pass/cap behavior, guard allow/deny, post-edit checks, and a full wf-run end-to-end (retry-with-feedback, crash sweep, rollback residue removal, topo order, blocking, labeled transcripts, pass commits). Point wf-run at the harness repo and it drives its own development.

Supporting layers:
- `PostToolUse` (`post-edit-check.sh`): instant syntax feedback on every Edit/Write (py/sh/json/yaml/toml) via exit-2 stderr, so errors surface at write time, not at the gate.
- `PreToolUse` (`pretool-guard.sh`): hard deny on destructive Bash patterns. A looped local model needs a floor.
- `SessionStart` (`session-context.sh`): injects branch, dirty files, TASKS.md head, and whether the gate is armed.
- `skills/task-loop`: TASKS.md-driven burn-down protocol; state lives on disk so context loss/compaction doesn't lose progress.
- `workflow-core.md` (single source, rendered into `skills/workflow/SKILL.md` and `opencode/agent/orchestrator.md`): the universal per-task state machine. States PLAN → IMPLEMENT → TEST → VERIFY → REVIEW → DOCUMENT → COMMIT with an explicit transition table: every gate failure returns to a defined earlier state carrying the blocker's verbatim structured report (`feedback` field), addressed first-item-first, then straight back to the failed gate. State persists in `.claude/wf-state.json` (whole-file rewrite per transition); gate hooks append matching records to `.claude/feedback.jsonl` (identical schema on both harnesses, no timestamps — deterministic). Escalation instead of spinning: 3 same-gate failures → task marked BLOCKED with final feedback, move on. Orchestration: in Claude Code the main session is the orchestrator (subagents can't spawn subagents), driven by the workflow skill; in opencode `orchestrator` is a primary-mode agent that @-delegates to the five subagents. `session-context.sh` injects wf-state + newest feedback at session start, so recovery after crash/compaction resumes mid-machine.
- `agents/` (mirrored in `opencode/agent/`): five fixed-format subagents — rigid report formats, since small models comply far better with them:
  - `planner`: goal → TASKS.md with per-item acceptance checks; flags verify.sh coverage gaps. Run first.
  - `qa-tester`: writes deterministic tests, reproduces bugs as failing tests, extends verify.sh; never fixes the code under test.
  - `verifier`: runs the suite, reports VERDICT/FAILURES/NEXT; never fixes.
  - `code-reviewer`: reviews the uncommitted diff; APPROVE/REVISE.
  - `doc-manager`: docstrings, module docs, README — derived strictly from code as-is; touches comments and .md only, never executable code; bugs it notices are reported, not fixed.

  Intended cycle per task: planner (once per goal) → implement → qa-tester → verifier → code-reviewer → doc-manager → commit. The Stop gate enforces the verifier's definition of done whether or not the agents are invoked.

## Sensor pack (harness-score L3+)
`project-template/` now ships the sensor and hygiene layer that maturity scorers (and more importantly, agents) need: `ruff.toml` (linter + formatter — deterministic per-edit feedback, and `post-edit-check.sh` now runs `ruff check` on every .py write when ruff is installed), `pyrightconfig.json` (standard-mode type checking — the compiler catches agent mistakes for free), `.pre-commit-config.yaml` (ruff/format/json/yaml/private-key checks pre-commit, quick verify pre-push), `.gitignore` (.env patterns plus harness state files that shouldn't be committed), `.mcp.json` (credentials via `${ENV_VAR}` interpolation, never literals), and `.claude/commands/` (task-loop, workflow, retro as explicit slash-command entry points). Drop these into any repo for the sensor items. `.github/workflows/ci.yml` runs all sensors on every push (CI-01/02/03), and `.github/workflows/harness.yml` is the L4 ratchet: `npx harness-score --min-level 4` fails any change that regresses the harness, plus badge generation. L4 requires L3 + hooks >= 70% + total >= 80% — with this pack applied the numbers clear comfortably; the ratchet is what makes the level durable rather than incidental. Note on scorers flagging `SessionStart`: it is a documented Claude Code hook event per the current hooks reference — prefer the working hook over the scorer point if a stale scorer disagrees.

## Install
```bash
mkdir -p ~/.claude/{hooks,agents,skills}
cp CLAUDE.md ~/.claude/CLAUDE.md
cp hooks/*.sh ~/.claude/hooks/ && chmod +x ~/.claude/hooks/*.sh
cp agents/*.md ~/.claude/agents/
cp -r skills/task-loop ~/.claude/skills/
# Merge settings.json into ~/.claude/settings.json (or copy if none exists)
```
Requires `jq` on PATH. Verify wiring with `/hooks` inside Claude Code.

Per project:
```bash
cp -r project-template/.claude <project>/
cp project-template/CLAUDE.md <project>/CLAUDE.md   # then fill in
chmod +x <project>/.claude/verify.sh
```
The template `verify.sh` exits 0 until you add checks; the gate only bites once it's real.

## opencode (full port in `opencode/`)
opencode has no declarative hook config; extension is a JS/TS plugin loaded from `~/.config/opencode/plugin/` (global) or `.opencode/plugin/` (project) that returns event handlers. `opencode/plugin/harness.js` implements the same three enforcement layers:

| Claude Code | opencode | Mechanism difference |
|---|---|---|
| Stop hook `decision:block` | `session.idle` handler | opencode idle handlers can't block the idle transition; on verify failure the plugin re-prompts the session via `client.session.prompt()` instead. Same loop, same cap (`OPENCODE_STOP_GATE_MAX`, default 8), plus subagent-skip via `parentID`. |
| PreToolUse deny | `tool.execute.before` (bash) | throwing an Error blocks the tool call |
| PostToolUse exit-2 | `tool.execute.before` (write) | content is syntax-checked pre-write via temp file (py/sh/json); edits are covered by the gate |

Install:
```bash
mkdir -p ~/.config/opencode/{plugin,agent,command}
cp opencode/plugin/harness.js ~/.config/opencode/plugin/
cp opencode/agent/*.md       ~/.config/opencode/agent/     # planner, qa-tester, verifier, code-reviewer, doc-manager
cp opencode/command/task-loop.md ~/.config/opencode/command/   # /task-loop
```
Rules: opencode natively reads `AGENTS.md`, but `opencode/opencode.json` shows the `instructions` config pointing at `CLAUDE.md`, so both harnesses share one rules file per project — merge that key into your `opencode.json`. The shared contract across both harnesses is exactly `.claude/verify.sh` + `TASKS.md` + `CLAUDE.md`; agents/commands/plugin are per-harness renderings of the same content.

Plugin was tested against mocked client/$ (guard block/pass, pre-write syntax block, gate fail→prompt with failure tail, pass→silent, 8-attempt cap, re-arm after pass, subagent skip).

## Model-side determinism (vLLM)
The gate makes the *outcome* deterministic (verifier-defined). For reproducible generations, also pin sampling in the serving layer: temperature 0 (greedy) or fixed seed, and note that batching/kernel nondeterminism can still vary outputs — the gate is the reliable layer, sampling config is best-effort.

## This time: version control
```bash
cd claude-harness && git init && git add -A && git commit -m "harness v1"
```
Symlink into place from the repo instead of copying if you want edits tracked live.
