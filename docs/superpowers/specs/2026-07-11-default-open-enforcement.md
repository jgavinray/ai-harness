# Default-open enforcement (spec amendment, 2026-07-11)

Amends the consistency design (2026-07-07) after the fifth live incident of
the same defect shape. Approved intent from the owner: "look at the system
holistically … really make the goals possible."

## The recurring defect shape

Every live failure of this harness to date has the same structure: an
**enumerated gate** — a list of permitted tools, permitted words, or
permitted steps — built from assumptions about what sessions look like,
denying legitimate behavior that fell outside the enumeration. The model
was never the problem; the fence was.

| date | gate | casualty |
| --- | --- | --- |
| pre-2026-07-08 | verify word-triggers (×3, eval-caught) | multi-step, find-and-report families |
| 2026-07-09 | plan status line bound 4 enforcement sites | first real workload unusable (55/58 requests locked) |
| 2026-07-09 | bash classifier missed `cd X && cmd` | project linter denied |
| 2026-07-11 | action-state tool allowlists omit `Agent` | subagent fan-out impossible; user's explicit "fan out subagents" denied 9+ times; two review sessions failed |

## The law (extends Law 6)

**Enforcement must be subtractive and evidence-cited.** A gate may only
*block* specific tools/actions whose misuse is named by a measured failure
(trace signature or eval regression in the gate's comment). A gate must
never enumerate the permitted surface: any enumeration silently denies
every tool, word, or workflow the author didn't foresee — including all
future client tools. Unknown ⇒ allowed; the client owns its tool surface.

Corollaries:

1. Action states carry `blocked_tools`, never `allowed_tools`. A state
   blocks only what it exists to prevent (verify blocks further edits until
   a check runs). `Agent`, `TodoWrite`, `AskUserQuestion`, and anything not
   yet invented pass through every state.
2. Read-before-edit lives in `guard_edit_without_read` (per-file, with
   actionable feedback), not in state tool-hiding — the state version was a
   coarser duplicate of an existing guard.
3. Catalog shaping follows the same rule: only verify state (the one
   deliberate pressure state) may subtract tools from the rendered catalog.
   This also removes a prefix-stability leak — the old inspect→edit_existing
   flip changed the rendered tool list mid-session and forced a re-prefill
   (Law 2).
4. **Telemetry is part of the gate.** Every denial is minable from traces;
   the nightly flywheel `gate_health` job aggregates denials per gate per
   day and writes `logs/gate_health/<date>.json`. A gate that fires on
   sessions that still end fine is a gate under suspicion. The next
   allowlist-vs-user fight must surface in a report the following morning,
   not in the owner's session weeks later.
5. The envelope only certifies workloads it contains: each live incident
   becomes an eval family (2026-07-09 → code-review; 2026-07-11 →
   review-fanout).

## Non-goals

Guards keyed on mechanical facts stay: unverified-edit verify pressure,
done-claim verification, loop detection, schema validation/repair, path
canon. Those bind on measured model failure modes, not on enumerations of
legitimate behavior.

## Eval gate

Full suite green + envelope re-run (all families supported) before this
ships as anything but a proposal.
