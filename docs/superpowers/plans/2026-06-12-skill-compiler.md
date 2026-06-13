# Skill Compiler Implementation Plan

**Status:** done 2026-06-13.

**Goal:** Compile installed skills into compact small-model procedures and make
runtime `Skill` calls inject the compiled procedure instead of requiring a large
raw skill body in the hot prompt.

## Tasks

- [x] Add `[skills]` config with installed-skill directory, cache directory, and
  max compiled token budget.
- [x] Add `SkillCompiler`, which reads `SKILL.md`, emits a numbered compact
  checklist, and caches it by skill name, model class, and content hash.
- [x] Add `scripts/compile_skills.py` for offline cache warming.
- [x] Intercept valid `Skill` tool calls in the relay, append the compiled
  procedure as feedback, and retry under the existing repair budget.
- [x] Record `skill_compiled` in relay metrics.
- [x] Add compiler/cache and relay-interception tests.

## Verification

Run:

```bash
.venv/bin/python -m pytest tests/test_skills.py tests/test_relay.py -q
.venv/bin/python -m pytest tests/ -q
```

Observed:

```text
16 passed
```
