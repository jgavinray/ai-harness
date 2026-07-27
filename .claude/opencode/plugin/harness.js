// harness.js — opencode port of the claude-harness hooks.
// Install: ~/.config/opencode/plugin/harness.js (global) or .opencode/plugin/harness.js (project).
//
// Parity with the Claude Code harness:
//   stop-gate.sh        -> session.idle handler: run .claude/verify.sh; on failure,
//                          re-prompt the session with the failure tail (iterate until green).
//                          Note: opencode idle handlers cannot BLOCK the idle transition;
//                          client.session.prompt() restarts the session instead. Same loop,
//                          different mechanism.
//   pretool-guard.sh    -> "tool.execute.before" on bash: throw to block destructive commands.
//   post-edit-check.sh  -> "tool.execute.before" on write: syntax-check content pre-write
//                          (best-effort; the verify gate is the enforcement layer).
//   Contract shared with Claude Code side: .claude/verify.sh + TASKS.md. No other coupling.

import { existsSync, writeFileSync, mkdtempSync, rmSync, appendFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, extname } from "node:path";

const MAX_ATTEMPTS = parseInt(process.env.OPENCODE_STOP_GATE_MAX ?? "8", 10);

const BLOCK_PATTERNS = [
  /rm\s+-rf\s+\/(?:[^a-zA-Z]|$)/,
  /rm\s+-rf\s+~/,
  /rm\s+-rf\s+"?\$HOME/,
  /mkfs\./,
  /dd\s+if=.*\s+of=\/dev\//,
  /git\s+push\s+[^\n]*--force/,
  /git\s+reset\s+--hard\s+origin/,
  /chmod\s+-R\s+777\s+\//,
  /\bshutdown\b/,
  /\breboot\b/,
];

// sessionID -> { attempts, running }
const gate = new Map();

async function runShell($, cmd) {
  // Bun shell: throws on nonzero exit; normalize to { ok, out }.
  try {
    const r = await $`bash -c ${cmd}`;
    return { ok: true, out: r?.stdout?.toString?.() ?? "" };
  } catch (e) {
    const out =
      (e?.stdout?.toString?.() ?? "") + (e?.stderr?.toString?.() ?? "") ||
      String(e);
    return { ok: false, out };
  }
}

function syntaxCheckCmd(ext, path) {
  switch (ext) {
    case ".py":
      return `python3 -m py_compile ${JSON.stringify(path)}`;
    case ".sh":
    case ".bash":
      return `bash -n ${JSON.stringify(path)}`;
    case ".json":
      return `jq empty ${JSON.stringify(path)}`;
    default:
      return null;
  }
}

export const Harness = async ({ client, $, directory }) => {
  const findVerify = () =>
    [join(directory, ".claude", "verify.sh"), join(directory, "verify.sh")].find(
      (p) => existsSync(p),
    );

  return {
    // ---- destructive-command floor ----------------------------------------
    "tool.execute.before": async (input, output) => {
      try {
        if (input.tool === "bash") {
          const cmd = String(output?.args?.command ?? "");
          if (BLOCK_PATTERNS.some((re) => re.test(cmd))) {
            throw new Error(
              `pretool-guard: destructive pattern blocked: ${cmd}. Use a targeted, reversible alternative.`,
            );
          }
          return;
        }

        // ---- pre-write syntax check (best-effort) --------------------------
        if (input.tool === "write") {
          const filePath = output?.args?.filePath ?? output?.args?.path;
          const content = output?.args?.content;
          if (typeof filePath !== "string" || typeof content !== "string") return;
          const cmdFor = syntaxCheckCmd(extname(filePath), "__TMP__");
          if (!cmdFor) return;
          const dir = mkdtempSync(join(tmpdir(), "oc-harness-"));
          const tmp = join(dir, "candidate" + extname(filePath));
          try {
            writeFileSync(tmp, content);
            const { ok, out } = await runShell($, cmdFor.replace("__TMP__", tmp));
            if (!ok) {
              throw new Error(
                `post-edit-check [${filePath}]: syntax error in proposed content:\n${out.slice(-1500)}`,
              );
            }
          } finally {
            rmSync(dir, { recursive: true, force: true });
          }
        }
      } catch (e) {
        if (e instanceof Error) throw e; // block the tool call
      }
    },

    // ---- iterate-until-done gate ------------------------------------------
    event: async ({ event }) => {
      if (event.type !== "session.idle") return;
      const sessionID = event.properties?.sessionID ?? event.properties?.sessionId;
      if (!sessionID) return;

      // Skip subagent sessions.
      try {
        const res = await client.session.get({ path: { id: sessionID } });
        const session = res?.data ?? res;
        if (session?.parentID) return;
      } catch {
        /* if lookup fails, proceed as main session */
      }

      const verify = findVerify();
      if (!verify) return;

      const st = gate.get(sessionID) ?? { attempts: 0, running: false };
      if (st.running) return;

      st.running = true;
      gate.set(sessionID, st);
      try {
        const { ok, out } = await runShell(
          $,
          `cd ${JSON.stringify(directory)} && timeout 600 bash ${JSON.stringify(verify)} 2>&1`,
        );
        if (ok) {
          st.cleared = true;
          return;
        }
        if (st.attempts >= MAX_ATTEMPTS) return; // capped: released; a future pass clears state
        st.attempts += 1;
        const tail = out.slice(-4000);
        // Durable feedback record for the workflow state machine. Same schema as
        // the Claude Code stop-gate. Deterministic fields only: no timestamps.
        try {
          appendFileSync(
            join(directory, ".claude", "feedback.jsonl"),
            JSON.stringify({
              source: "stop-gate",
              session: sessionID,
              attempt: st.attempts,
              exit: 1,
              report: tail,
            }) + "\n",
          );
        } catch {
          /* feedback log is best-effort */
        }
        await client.session.prompt({
          path: { id: sessionID },
          body: {
            parts: [
              {
                type: "text",
                text:
                  `verify.sh FAILED (attempt ${st.attempts}/${MAX_ATTEMPTS}). ` +
                  `You are not done. Fix the first failure below, then re-run ` +
                  `.claude/verify.sh yourself before finishing. Output tail:\n${tail}`,
              },
            ],
          },
        });
      } finally {
        st.running = false;
        if (st.cleared) gate.delete(sessionID);
        else gate.set(sessionID, st);
      }
    },
  };
};
