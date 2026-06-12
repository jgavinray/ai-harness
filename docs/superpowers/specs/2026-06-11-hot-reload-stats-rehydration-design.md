# Hot reload + stats rehydration

Date: 2026-06-11. Approved scope: both features, selected by user.

## Problem

`create_app` reads `harness.toml` once; any config change (e.g. fleet role
reassignment) requires a restart. A restart zeroes the in-memory `/stats`
aggregates (global input/output/cached token totals and per-backend
counters), drops the router's session-affinity map (cooling backend KV
caches), and empties the response cache. The per-request detail is already
durable in `logs/requests.jsonl`; only the aggregates are lost.

## Feature 1: `POST /admin/reload`

Re-reads the config file the server was started with and reconfigures the
backend fleet **in place**, preserving all counters and affinity.

- `create_app(settings, backend_client=None, config_path=None)` gains a
  `config_path` parameter; `__main__` passes `args.config`.
- `BackendPool.reconfigure(settings)` diffs the new `[[backends]]` list
  against the live pool by backend `name`:
  - **surviving name**: update `cfg`, `roles`, rebuilt transport/profile;
    keep `requests`, `errors`, `prompt_tokens`, `cached_tokens`,
    `ttft_ms`, `in_flight`, breaker state.
  - **new name**: append a fresh `PooledBackend`.
  - **missing name**: drop from the pool (in-flight streams hold their own
    reference and finish; `Router.pick` already tolerates affinity entries
    pointing at removed backends).
  - returns `{"updated": [...], "added": [...], "removed": [...]}`.
- Endpoint returns 400 if the server was built without a `config_path`.
- **Scope: `[[backends]]` only.** `[server]`, `[pipeline]`, `[cache]`,
  `[memory]`, `[traces]`, `[log]` still need a restart; the endpoint says
  so in its response. No SIGHUP handler (YAGNI — curl covers it).

## Feature 2: stats rehydration on startup

When `log.requests_path` is set and the file exists, `create_app` replays
it once to seed the aggregates, so a restart no longer zeroes
input/output/cached totals.

- Global stats: `requests` +1 per record, `errors` +1 when the record has
  an `error` key, sum `input_tokens` / `output_tokens` / `cached_tokens`.
- Per backend (matched by record `backend` name; unknown names skipped):
  mirror live counting — `requests` +1 except for response-cache hits
  (`cache == "response"`), `errors` +1 on `error`, sum `prompt_tokens` /
  `cached_tokens`, append `ttft_ms` capped to the rolling `TTFT_WINDOW`.
- Malformed lines are skipped.

## Tests

- Pool: reconfigure updates roles in place and preserves counters;
  reconfigure adds new and drops removed backends.
- Server: `POST /admin/reload` applies edited roles from the toml without
  resetting `/stats` counters; 400 without a config path.
- Server: startup with a populated `requests.jsonl` reports the replayed
  totals in `/stats`.
