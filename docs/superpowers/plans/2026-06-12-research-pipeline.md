# Research Pipeline Implementation Plan

**Status:** done 2026-06-13.

**Goal:** Give small models a codified research path: fetch source material,
summarize through a cheap backend, cache the resulting brief, and inject only the
brief into the main conversation.

## Tasks

- [x] Add `[research]` config with cache directory and source/chunk budgets.
- [x] Detect explicit `research: <source>` user requests.
- [x] Fetch `file://`, `http://`, and `https://` sources, then chunk source text.
- [x] Summarize chunks through a `research` role backend when available, falling
  back to `fast` and then any backend.
- [x] Cache briefs by query hash and inject `## Research brief` into the system
  prompt; raw retrieval is not injected.
- [x] Report `research_briefs` in request/eval metrics.
- [x] Add server tests for generation, caching, and prompt injection.

## Verification

Run:

```bash
.venv/bin/python -m pytest tests/test_server.py::test_research_brief_generated_cached_and_injected tests/test_evals.py -q
.venv/bin/python -m pytest tests/ -q
```

Observed:

```text
6 passed
175 passed
```
