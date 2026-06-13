# Growth Loop + Candidate Adoption Implementation Plan

**Status:** done 2026-06-13.

**Goal:** Close the non-serving side of the platform loop: traces become gated
corpus rows, candidate backends are evaluated without live routing, and promotion
is an auditable config change.

## Tasks

- [x] Keep `roles = ["candidate"]` backends out of live role routing, fallback
  routing, and capability routing.
- [x] Add `scripts/shadow_eval.py` to generate eval commands for every candidate
  backend in a harness config.
- [x] Add `scripts/promote_candidate.py` to compare incumbent/candidate eval
  success rates and update backend roles only when the gate passes.
- [x] Add `scripts/lora_train.py` as a LoRA training job scaffold over the gated
  corpus produced by `scripts/corpus.py`.
- [x] Document candidate roles in `harness.toml.example`.
- [x] Add routing and script tests.

## Verification

Run:

```bash
.venv/bin/python -m pytest tests/test_pool_router.py tests/test_growth.py -q
.venv/bin/python -m pytest tests/ -q
```

Observed:

```text
30 passed
180 passed
```
