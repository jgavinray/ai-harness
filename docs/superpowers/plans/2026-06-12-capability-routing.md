# Capability Routing + OCR Fallback Implementation Plan

**Status:** done 2026-06-13.

**Goal:** Let backends declare capability tags and route modality-specific
requests to capable backends, with a codified text fallback when no vision backend
exists.

## Tasks

- [x] Add `capabilities: list[str]` to `PoolBackendCfg`.
- [x] Add request inspection for image blocks and expose it as
  `request_capabilities`.
- [x] Route image requests to a backend tagged with `vision` when one is live.
- [x] Preserve one-endpoint behavior when no vision backend exists by replacing
  image blocks with a framed text fallback before decoding.
- [x] Log `capability_fallbacks` in request metrics and eval reports.
- [x] Document backend capability tags in `harness.toml.example`.
- [x] Add router and server logging tests for capability selection and fallback.

## Verification

Run:

```bash
.venv/bin/python -m pytest tests/test_pool_router.py tests/test_server_logging.py tests/test_evals.py -q
.venv/bin/python -m pytest tests/ -q
```

Observed:

```text
34 passed
174 passed
```
