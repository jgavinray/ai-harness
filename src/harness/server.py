"""FastAPI app: the Anthropic Messages API surface Claude Code talks to.

Every response path — success, backend failure, bad request — produces a
spec-valid Anthropic response so Claude Code's own retry/UX logic works.
Requests are routed across the backend fleet with session affinity; fast-
role responses are served from the exact-match response cache when possible.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

DASHBOARD = Path(__file__).parent / "static" / "dashboard.html"

from harness import relay
from harness.backends.base import BackendError
from harness.backends.pool import BackendPool
from harness.cache import ResponseCache, payload_key
from harness.codec.anthropic_in import decode
from harness.codec.anthropic_out import collect, error_body, error_sse, stream_sse
from harness.config import Settings, load_settings
from harness.ir import Done
from harness.log import RequestLogger
from harness.memory import MemoryManager, MemoryStage
from harness.pipeline.base import run_pipeline
from harness.pipeline.fewshot import FewshotStage
from harness.pipeline.history import HistoryStage
from harness.pipeline.system_prompt import SystemPromptStage
from harness.pipeline.tool_prune import ToolPruneStage
from harness.pipeline.tool_schema import ToolSchemaStage
from harness.router import Router, session_key
from harness.tokens.counter import HeuristicCounter, count_conversation
from harness.traces import TraceStore

STAGES = [
    SystemPromptStage(),
    ToolPruneStage(),
    ToolSchemaStage(),
    HistoryStage(),
    FewshotStage(),
]
TTFT_WINDOW = 500
CACHE_WINDOW = 100  # requests in the rolling kv-hit window

# Prometheus gauge (0..1) for live KV pool occupancy, per backend kind.
# llama.cpp only serves /metrics when launched with --metrics.
KV_USAGE_GAUGES = {
    "vllm": "vllm:kv_cache_usage_perc",
    "llamacpp": "llamacpp:kv_cache_usage_ratio",
}


async def _kv_usage(b, client: httpx.AsyncClient) -> float | None:
    gauge = KV_USAGE_GAUGES.get(b.cfg.kind)
    if not gauge:
        return None
    url = b.cfg.base_url.rstrip("/").removesuffix("/v1") + "/metrics"
    try:
        resp = await client.get(url, timeout=2.0)
        if resp.status_code != 200:
            return None
        for line in resp.text.splitlines():
            if line.startswith(gauge):
                return round(float(line.rsplit(None, 1)[-1]) * 100, 1)
    except (httpx.HTTPError, ValueError):
        return None
    return None


def _dump(settings: Settings, kind: str, data: dict) -> None:
    if not settings.debug.dump_prompts:
        return
    d = Path(settings.debug.dump_dir)
    d.mkdir(parents=True, exist_ok=True)
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}-{kind}.json"
    (d / name).write_text(json.dumps(data, indent=2))


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * pct), len(ordered) - 1)]


async def _replay(events):
    for ev in events:
        yield ev


def _seed_stats(stats: dict, pool: BackendPool, path: str | Path) -> None:
    """Replay the request log so a restart doesn't zero /stats aggregates.

    Mirrors live counting: response-cache hits never reached the backend, so
    they skip the backend request counter but still carry their token usage.
    """
    p = Path(path)
    if not p.exists():
        return
    with p.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            stats["requests"] += 1
            if r.get("error"):
                stats["errors"] += 1
            stats["input_tokens"] += r.get("input_tokens") or 0
            stats["output_tokens"] += r.get("output_tokens") or 0
            stats["cached_tokens"] += r.get("cached_tokens") or 0
            b = pool.get(r.get("backend") or "")
            if b is None:
                continue
            if r.get("cache") != "response":
                b.requests += 1
            if r.get("error"):
                b.errors += 1
            b.prompt_tokens += r.get("input_tokens") or 0
            b.cached_tokens += r.get("cached_tokens") or 0
            b.output_tokens += r.get("output_tokens") or 0
            b.recent_cache.append((r.get("input_tokens") or 0, r.get("cached_tokens") or 0))
            if r.get("ttft_ms") is not None:
                b.ttft_ms.append(r["ttft_ms"])
    for b in pool.backends:
        del b.ttft_ms[:-TTFT_WINDOW]
        del b.recent_cache[:-CACHE_WINDOW]


def create_app(
    settings: Settings,
    backend_client: httpx.AsyncClient | None = None,
    config_path: str | Path | None = None,
) -> FastAPI:
    app = FastAPI(title="ai-harness")
    client = backend_client or httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10))
    pool = BackendPool(settings, client)
    router = Router(pool, settings)
    rcache = ResponseCache(settings.cache.ttl_s, settings.cache.max_entries)
    counter = HeuristicCounter()
    logger = RequestLogger(settings.log.requests_path)
    traces = TraceStore(settings.traces.dir if settings.traces.enabled else None)

    async def fast_complete(messages: list[dict]) -> str:
        candidates = pool.with_role("fast") or pool.backends
        b = min(candidates, key=lambda x: x.in_flight)
        payload = {
            "model": b.model_name, "messages": messages, "max_tokens": 400,
            "stream": True, "stream_options": {"include_usage": True},
        }
        from harness.ir import TextDelta
        text = ""
        async for ev in b.profile.parse(b.stream(payload)):
            if isinstance(ev, TextDelta):
                text += ev.text
        return text

    memory = MemoryManager(settings, fast_complete if settings.memory.enabled else None)
    stages = STAGES + [MemoryStage(memory, settings)]
    stats = {"requests": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0,
             "cached_tokens": 0}
    if settings.log.requests_path:
        _seed_stats(stats, pool, settings.log.requests_path)

    def invalid_request(message: str) -> JSONResponse:
        return JSONResponse(error_body("invalid_request_error", message), status_code=400)

    @app.post("/v1/messages")
    async def messages(request: Request):
        try:
            body = await request.json()
        except Exception:
            return invalid_request("body is not valid JSON")
        if "messages" not in body or "max_tokens" not in body:
            return invalid_request("'messages' and 'max_tokens' are required")
        try:
            conv = decode(body)
        except (KeyError, TypeError, AttributeError) as exc:
            return invalid_request(f"could not decode request: {exc!r}")

        _dump(settings, "anthropic-request", body)
        chosen = router.pick(body)
        # The compaction budget depends on which backend serves the request,
        # so route first and pipeline against that backend's context window.
        req_settings = settings.model_copy(deep=True)
        req_settings.profile.context_window = chosen.cfg.context_window
        conv = run_pipeline(conv, req_settings, stages)
        skey = session_key(body)
        role = "fast" if "haiku" in (body.get("model") or "") else "main"
        rendered = chosen.profile.render(conv, chosen.model_name)
        _dump(settings, "rendered-payload", rendered)

        stats["requests"] += 1
        msg_id = "msg_" + uuid.uuid4().hex[:24]
        model = body.get("model", chosen.model_name)
        metrics: dict = {}
        record: dict = {
            "request_id": msg_id,
            "session_key": skey,
            "model": model,
            "backend": chosen.name,
            "role": role,
            "stream": conv.params.stream,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "stop_reason": None,
            "ttft_ms": None,
        }
        start = time.monotonic()

        cacheable = settings.cache.enabled and role in settings.cache.roles
        cache_key = payload_key(rendered) if cacheable else None
        cached_events = rcache.get(cache_key) if cache_key else None
        if cached_events is not None:
            record["cache"] = "response"
            events = _replay(cached_events)
        else:
            events = relay.run(conv, chosen.profile, chosen, settings, metrics=metrics)

        buffer: list = []

        async def _instrument(evs):
            async for ev in evs:
                if record["ttft_ms"] is None:
                    ttft = int((time.monotonic() - start) * 1000)
                    record["ttft_ms"] = ttft
                    chosen.ttft_ms.append(ttft)
                    del chosen.ttft_ms[:-TTFT_WINDOW]
                if isinstance(ev, Done):
                    record["input_tokens"] = ev.input_tokens
                    record["output_tokens"] = ev.output_tokens
                    record["cached_tokens"] = ev.cached_tokens
                    record["stop_reason"] = ev.stop_reason
                    stats["input_tokens"] += ev.input_tokens
                    stats["output_tokens"] += ev.output_tokens
                    stats["cached_tokens"] += ev.cached_tokens
                    chosen.prompt_tokens += ev.input_tokens
                    chosen.cached_tokens += ev.cached_tokens
                    chosen.output_tokens += ev.output_tokens
                    chosen.recent_cache.append((ev.input_tokens, ev.cached_tokens))
                    del chosen.recent_cache[:-CACHE_WINDOW]
                if (cache_key and cached_events is None) or settings.traces.enabled:
                    buffer.append(ev)
                yield ev

        def _finish_record() -> None:
            record["wall_ms"] = int((time.monotonic() - start) * 1000)
            if record["ttft_ms"] is None:
                record["ttft_ms"] = record["wall_ms"]
            record.update(metrics)
            if (
                cache_key
                and cached_events is None
                and "error" not in record
                and buffer
                and isinstance(buffer[-1], Done)
            ):
                rcache.put(cache_key, buffer)
            if settings.traces.enabled and cached_events is None:
                traces.append(skey, msg_id, rendered, buffer, dict(metrics))
            if settings.memory.enabled and role == "main":
                memory.note(skey, conv.system, rendered.get("messages", []))
                try:
                    asyncio.get_running_loop().create_task(memory.sweep())
                except RuntimeError:
                    pass
            logger.write(record)

        if conv.params.stream:
            async def sse():
                try:
                    async for piece in stream_sse(_instrument(events), model, msg_id):
                        yield piece
                except BackendError as exc:
                    stats["errors"] += 1
                    record["error"] = str(exc)
                    yield error_sse("overloaded_error", str(exc))
                finally:
                    _finish_record()

            return StreamingResponse(sse(), media_type="text/event-stream")

        try:
            collected = [e async for e in _instrument(events)]
        except BackendError as exc:
            stats["errors"] += 1
            record["error"] = str(exc)
            _finish_record()
            return JSONResponse(error_body("overloaded_error", str(exc)), status_code=529)
        _finish_record()
        return JSONResponse(collect(collected, model, msg_id))

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        try:
            body = await request.json()
            conv = decode(body)
        except Exception as exc:
            return invalid_request(f"could not decode request: {exc!r}")
        return JSONResponse({"input_tokens": count_conversation(conv, counter)})

    @app.get("/dashboard")
    async def dashboard():
        return HTMLResponse(DASHBOARD.read_text())

    @app.post("/admin/reload")
    async def admin_reload():
        if not config_path:
            return JSONResponse(
                {"error": "server was started without a config file; nothing to reload"},
                status_code=400,
            )
        summary = pool.reconfigure(load_settings(config_path))
        return JSONResponse(
            {"scope": "backends only; other sections need a restart", **summary}
        )

    @app.get("/stats")
    async def get_stats():
        usages = await asyncio.gather(*(_kv_usage(b, client) for b in pool.backends))
        backends = {}
        for b, kv_used in zip(pool.backends, usages):
            total_prompt = b.prompt_tokens or 1
            recent_prompt = sum(p for p, _ in b.recent_cache) or 1
            recent_cached = sum(c for _, c in b.recent_cache)
            backends[b.name] = {
                "model": b.model_name,
                "roles": b.roles,
                "requests": b.requests,
                "errors": b.errors,
                "in_flight": b.in_flight,
                "down": b.is_down(),
                "ttft_p50_ms": _percentile(b.ttft_ms, 0.50),
                "ttft_p95_ms": _percentile(b.ttft_ms, 0.95),
                "kv_cache_hit_pct": round(100 * b.cached_tokens / total_prompt, 1),
                "kv_cache_hit_pct_recent": round(100 * recent_cached / recent_prompt, 1),
                # fresh prefill + every decoded token = all tokens written to KV
                "kv_written_tokens": b.prompt_tokens - b.cached_tokens + b.output_tokens,
                "kv_used_pct": kv_used,
            }
        return JSONResponse({
            **stats,
            "backends": backends,
            "response_cache": {"hits": rcache.hits, "misses": rcache.misses},
        })

    return app
