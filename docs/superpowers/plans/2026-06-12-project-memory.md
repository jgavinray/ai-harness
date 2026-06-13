# Project Memory Implementation Plan

**Status:** done 2026-06-13.

**Goal:** Complete the project-memory tier from the platform spec: durable
per-repo facts are distilled from traces off the serving path and injected
byte-stably at session start.

## Tasks

- [x] Keep runtime injection through `MemoryStage`, bounded by
  `memory.max_chars` and framed under `## Project memory (from previous sessions)`.
- [x] Add `scripts/memory_distill.py`, an offline trace distiller that reads
  `traces/sessions.jsonl`, skips mechanically dirty rows, and merges durable
  project facts into the existing memory store.
- [x] Extract low-risk deterministic facts first: project key and verified Bash
  commands from clean traces.
- [x] Add `memory_tokens` request logging so memory cost is visible in
  `logs/requests.jsonl`.
- [x] Add tests for offline distillation, injected memory token counting, and
  request logging.

## Verification

Run:

```bash
.venv/bin/python -m pytest tests/test_memory.py tests/test_server_logging.py -q
.venv/bin/python -m pytest tests/ -q
```

Observed:

```text
10 passed
169 passed
```
