# Correction inbox

Verbatim user corrections, appended as they happen. These outlive the session
that produced them. Do not paraphrase; do not delete without the user saying so.

---

## 2026-07-30 — no GPG-signed commits

> There is to be no commits with gpg signing - this is a hard rule

**Context:** committing the default-open MCP surface fix. The maintainer's
GLOBAL git config sets `commit.gpgsign = true`, so `git commit` launched
pinentry and failed with "gpg: signing failed: Operation cancelled".

**Applied:** `git config --local commit.gpgsign false` (and `tag.gpgsign
false`) in this repo, so the rule holds for every tool operating here, not
just one command. The global setting was left untouched. Commits are also
made with an explicit `--no-gpg-sign`.

**Related:** `evals/run.py` had the same class of bug from the other
direction — its throwaway `eval <eval@local>` identity inherited global
signing and had no key, killing every eval trial at repo seeding. Fixed the
same day with `-c commit.gpgsign=false`.
