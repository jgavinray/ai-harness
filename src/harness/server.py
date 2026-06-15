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
from dataclasses import replace
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

DASHBOARD = Path(__file__).parent / "static" / "dashboard.html"

from harness import ocr, relay
from harness.backends.base import BackendError
from harness.backends.pool import BackendPool
from harness.cache import ResponseCache, payload_key
from harness.codec.anthropic_in import decode
from harness.codec.anthropic_out import collect, error_body, error_sse, stream_sse
from harness.config import Settings, load_settings
from harness.critic import CriticManager
from harness.ir import Done, ThinkingDelta, ToolResultPart
from harness.log import RequestLogger
from harness.memory import MemoryManager, MemoryStage, injected_memory_tokens, project_key
from harness.planning import PlanningManager
from harness.review import ReviewManager
from harness.research import ResearchManager, memory_fact
from harness.reasoning_budget import apply_reasoning_budget
from harness.pipeline.base import run_pipeline
from harness.pipeline.fewshot import FewshotStage
from harness.pipeline.history import HistoryStage
from harness.pipeline.path_canon import PathCanonStage
from harness.pipeline.system_prompt import SystemPromptStage
from harness.pipeline.tool_prune import ToolPruneStage
from harness.pipeline.tool_schema import ToolSchemaStage
from harness.router import Router, request_capabilities, request_role, session_key
from harness.tokens.counter import HeuristicCounter, count_conversation
from harness.traces import TraceStore

STAGES = [
    PathCanonStage(),
    SystemPromptStage(),
    ToolPruneStage(),
    ToolSchemaStage(),
    HistoryStage(),
    FewshotStage(),
]
TTFT_WINDOW = 500
CACHE_WINDOW = 100  # requests in the rolling kv-hit window
CRITIC_WINDOW = 100
KV_USED_TTL_S = 60.0  # how long a missed poll may serve the last good reading

# Prometheus gauge (0..1) for live KV pool occupancy, per backend kind.
# llama.cpp only serves /metrics when launched with --metrics.
KV_USAGE_GAUGES = {
    "vllm": "vllm:kv_cache_usage_perc",
    "llamacpp": "llamacpp:kv_cache_usage_ratio",
}


KV_RESIDENT_SESSIONS = 8  # sessions remembered per backend for the estimate


def _apply_relaxed(settings: Settings, relaxed: list[str]) -> None:
    """Disable eval-retired scaffolds for the backend handling this request."""
    for item in relaxed:
        if item == "workflow_guards":
            settings.pipeline.workflow_guards = False
        elif item == "guard_edit_without_read":
            settings.pipeline.guard_edit_without_read = False
        elif item == "guard_verify_after_edit":
            settings.pipeline.guard_verify_after_edit = False
        elif item in {"planning", "planning_scaffold"}:
            settings.planning.enabled = False
        elif item == "skills":
            settings.skills.enabled = False
        elif item == "research":
            settings.research.enabled = False
        elif item == "tool_catalog":
            settings.pipeline.tool_catalog = False
        elif item == "fewshot":
            settings.pipeline.fewshot = False


async def _kv_usage(b, client: httpx.AsyncClient) -> dict | None:
    """Live KV occupancy: measured from the engine's gauge when it has one,
    otherwise (modern llama.cpp dropped its KV metrics) estimated from slot
    capacity (/slots n_ctx) and the sessions most recently resident here."""
    gauge = KV_USAGE_GAUGES.get(b.cfg.kind)
    if not gauge:
        return None
    base = b.cfg.base_url.rstrip("/").removesuffix("/v1")
    slots = max(1, b.cfg.max_in_flight or b.in_flight or 1)
    resident_estimate = sum(list(b.kv_resident.values())[-slots:])
    estimated_pct = round(100 * min(resident_estimate, b.cfg.context_window * slots) / (b.cfg.context_window * slots), 1)
    try:
        resp = await client.get(base + "/metrics", timeout=2.0)
        if resp.status_code == 200:
            for line in resp.text.splitlines():
                if line.startswith(gauge):
                    measured_pct = round(float(line.rsplit(None, 1)[-1]) * 100, 1)
                    if b.cfg.kind == "vllm" and estimated_pct > measured_pct:
                        return {"pct": estimated_pct, "est": True}
                    return {"pct": measured_pct, "est": False}
    except (httpx.HTTPError, ValueError):
        pass
    if b.cfg.kind != "llamacpp":
        return None
    try:
        resp = await client.get(base + "/slots", timeout=2.0)
        if resp.status_code != 200:
            return None
        slots = resp.json()
        capacity = sum(s.get("n_ctx") or 0 for s in slots)
        if not slots or not capacity:
            return None
        estimated = sum(list(b.kv_resident.values())[-len(slots):])
        # busy slots report real token counts; never report below them
        live = sum(s.get("n_prompt_tokens") or 0 for s in slots)
        resident = max(estimated, live)
        return {"pct": round(100 * min(resident, capacity) / capacity, 1), "est": True}
    except (httpx.HTTPError, ValueError):
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


def _readonly_reasoning(conv, settings: Settings):
    allowed = set(settings.routing.reasoning_readonly_tools)
    tools = tuple(t for t in conv.tools if t.name in allowed)
    return replace(conv, tools=tools, all_tools=tools)


async def _replay(events):
    for ev in events:
        yield ev


def _empty_critic_stats() -> dict:
    return {
        "calls": 0,
        "approve": 0,
        "revise": 0,
        "inconclusive": 0,
        "errors": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens_observed": 0,
        "triggers": {},
        "profiles": {},
        "feedback_tags": {},
        "feedback_hashes": {},
        "inconclusive_reasons": {},
        "recent": [],
    }


def _empty_runtime_stats() -> dict:
    return {
        "valid_calls": 0,
        "invalid_calls": 0,
        "preflight_rewrites": 0,
        "preflight_denies": 0,
        "preflight_reasons": {},
        "tool_success_after_preflight": 0,
        "tool_failure_after_preflight": 0,
        "context_compactions": 0,
        "turns_elided": 0,
        "tool_results_truncated": 0,
        "context_samples": 0,
        "context_tokens_before_total": 0,
        "context_tokens_after_total": 0,
        "latest_context_tokens_before": None,
        "latest_context_tokens_after": None,
        "action_state_blocks": 0,
        "critic_skips": {},
        "critic_saved_turn_estimate": 0,
        "tool_count_samples": 0,
        "client_tools_total": 0,
        "pipeline_tools_total": 0,
        "backend_tools_total": 0,
        "latest_client_tool_count": None,
        "latest_pipeline_tool_count": None,
        "latest_backend_tool_count": None,
        "max_client_tool_count": 0,
        "max_pipeline_tool_count": 0,
        "max_backend_tool_count": 0,
        "client_tool_count_hist": {},
        "pipeline_tool_count_hist": {},
        "backend_tool_count_hist": {},
    }


def _inc_counter(target: dict, key: str | None, amount: int = 1) -> None:
    if not key:
        return
    target[key] = target.get(key, 0) + amount


def _record_critic_stats(critic_stats: dict, record: dict) -> None:
    if record.get("sidecar_type") != "critic":
        return
    critic_stats["calls"] += 1
    action = record.get("critic_action")
    if action == "approve":
        critic_stats["approve"] += 1
    elif action == "revise":
        critic_stats["revise"] += 1
    elif action == "inconclusive":
        critic_stats["inconclusive"] += 1
    if record.get("error"):
        critic_stats["errors"] += 1
    critic_stats["input_tokens"] += record.get("input_tokens") or 0
    critic_stats["output_tokens"] += record.get("output_tokens") or 0
    critic_stats["reasoning_tokens_observed"] += record.get("reasoning_tokens_observed") or 0
    for trigger in record.get("critic_triggers") or []:
        _inc_counter(critic_stats["triggers"], trigger)
    for profile in record.get("critic_matched_profiles") or []:
        _inc_counter(critic_stats["profiles"], profile)
    for tag in record.get("critic_feedback_tags") or []:
        _inc_counter(critic_stats["feedback_tags"], tag)
    if record.get("critic_feedback_hash"):
        _inc_counter(critic_stats["feedback_hashes"], record["critic_feedback_hash"])
    _inc_counter(critic_stats["inconclusive_reasons"], record.get("critic_inconclusive_reason"))
    critic_stats["recent"].append({
        "action": action,
        "triggers": record.get("critic_triggers") or [],
        "profiles": record.get("critic_matched_profiles") or [],
        "feedback_tags": record.get("critic_feedback_tags") or [],
        "feedback_hash": record.get("critic_feedback_hash"),
        "inconclusive_reason": record.get("critic_inconclusive_reason"),
        "parent_request_id": record.get("parent_request_id"),
    })
    del critic_stats["recent"][:-CRITIC_WINDOW]


def _critic_summary(critic_stats: dict) -> dict:
    recent = critic_stats["recent"]
    recent_revise = sum(1 for r in recent if r.get("action") == "revise")
    recent_inconclusive = sum(1 for r in recent if r.get("action") == "inconclusive")
    repeated = {
        k: v for k, v in critic_stats["feedback_hashes"].items()
        if v > 1
    }
    return {
        **{k: v for k, v in critic_stats.items() if k != "recent"},
        "revise_pct": round(100 * critic_stats["revise"] / (critic_stats["calls"] or 1), 1),
        "recent_calls": len(recent),
        "recent_revise": recent_revise,
        "recent_revise_pct": round(100 * recent_revise / (len(recent) or 1), 1),
        "recent_inconclusive": recent_inconclusive,
        "recent_inconclusive_pct": round(100 * recent_inconclusive / (len(recent) or 1), 1),
        "repeated_feedback_hashes": repeated,
    }


def _record_runtime_stats(runtime_stats: dict, record: dict) -> None:
    runtime_stats["valid_calls"] += record.get("valid_calls") or 0
    runtime_stats["invalid_calls"] += record.get("invalid_calls") or 0
    runtime_stats["preflight_rewrites"] += record.get("preflight_rewrites") or 0
    runtime_stats["preflight_denies"] += record.get("preflight_denies") or 0
    runtime_stats["tool_success_after_preflight"] += record.get("tool_success_after_preflight") or 0
    runtime_stats["tool_failure_after_preflight"] += record.get("tool_failure_after_preflight") or 0
    for reason, count in (record.get("preflight_reasons") or {}).items():
        _inc_counter(runtime_stats["preflight_reasons"], reason, count)
    if record.get("context_compacted"):
        runtime_stats["context_compactions"] += 1
    runtime_stats["turns_elided"] += record.get("turns_elided") or 0
    runtime_stats["tool_results_truncated"] += record.get("tool_results_truncated") or 0
    if record.get("context_tokens_before") is not None:
        runtime_stats["context_samples"] += 1
        runtime_stats["context_tokens_before_total"] += record.get("context_tokens_before") or 0
        runtime_stats["context_tokens_after_total"] += record.get("context_tokens_after") or 0
        runtime_stats["latest_context_tokens_before"] = record.get("context_tokens_before")
        runtime_stats["latest_context_tokens_after"] = record.get("context_tokens_after")
    runtime_stats["action_state_blocks"] += record.get("action_state_blocks") or 0
    _inc_counter(runtime_stats["critic_skips"], record.get("critic_skipped_reason"))
    runtime_stats["critic_saved_turn_estimate"] += record.get("critic_saved_turn_estimate") or 0
    if record.get("client_tool_count") is not None:
        runtime_stats["tool_count_samples"] += 1
        for prefix, key in (
            ("client", "client_tool_count"),
            ("pipeline", "pipeline_tool_count"),
            ("backend", "backend_tool_count"),
        ):
            value = record.get(key)
            if value is None:
                continue
            runtime_stats[f"{prefix}_tools_total"] += value
            runtime_stats[f"latest_{key}"] = value
            runtime_stats[f"max_{key}"] = max(runtime_stats[f"max_{key}"], value)
            _inc_counter(runtime_stats[f"{key}_hist"], str(value))


def _runtime_summary(runtime_stats: dict) -> dict:
    total_tool_calls = runtime_stats["valid_calls"] + runtime_stats["invalid_calls"]
    invalid_rate = round(100 * runtime_stats["invalid_calls"] / (total_tool_calls or 1), 1)
    tool_samples = runtime_stats["tool_count_samples"] or 1
    return {
        **runtime_stats,
        "invalid_tool_rate_pct": invalid_rate,
        "client_tool_count_avg": round(runtime_stats["client_tools_total"] / tool_samples, 1),
        "pipeline_tool_count_avg": round(runtime_stats["pipeline_tools_total"] / tool_samples, 1),
        "backend_tool_count_avg": round(runtime_stats["backend_tools_total"] / tool_samples, 1),
        "context_tokens_before_avg": round(
            runtime_stats["context_tokens_before_total"] / (runtime_stats["context_samples"] or 1)
        ),
        "context_tokens_after_avg": round(
            runtime_stats["context_tokens_after_total"] / (runtime_stats["context_samples"] or 1)
        ),
    }


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
            if "critic" in stats:
                _record_critic_stats(stats["critic"], r)
            if "runtime" in stats:
                _record_runtime_stats(stats["runtime"], r)
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
            skey = r.get("session_key")
            if skey and (r.get("input_tokens") or 0):
                b.kv_resident.pop(skey, None)
                b.kv_resident[skey] = (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0)
                while len(b.kv_resident) > KV_RESIDENT_SESSIONS:
                    b.kv_resident.pop(next(iter(b.kv_resident)))
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
    planner = PlanningManager(settings)
    reviewer = ReviewManager(settings)
    critic = CriticManager(settings)
    research = ResearchManager(settings)
    pending_preflight: dict[tuple[str, str], dict] = {}

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
             "cached_tokens": 0, "critic": _empty_critic_stats(),
             "runtime": _empty_runtime_stats()}
    if settings.log.requests_path:
        _seed_stats(stats, pool, settings.log.requests_path)

    def account_usage(
        b,
        done: Done,
        skey: str | None = None,
        *,
        count_request: bool = False,
        ttft_ms: int | None = None,
    ) -> None:
        if count_request:
            stats["requests"] += 1
        if ttft_ms is not None:
            b.ttft_ms.append(ttft_ms)
            del b.ttft_ms[:-TTFT_WINDOW]
        stats["input_tokens"] += done.input_tokens
        stats["output_tokens"] += done.output_tokens
        stats["cached_tokens"] += done.cached_tokens
        b.prompt_tokens += done.input_tokens
        b.cached_tokens += done.cached_tokens
        b.output_tokens += done.output_tokens
        b.recent_cache.append((done.input_tokens, done.cached_tokens))
        del b.recent_cache[:-CACHE_WINDOW]
        if skey and done.input_tokens:
            b.kv_resident.pop(skey, None)
            b.kv_resident[skey] = done.input_tokens + done.output_tokens
            while len(b.kv_resident) > KV_RESIDENT_SESSIONS:
                b.kv_resident.pop(next(iter(b.kv_resident)))

    def invalid_request(message: str) -> JSONResponse:
        return JSONResponse(error_body("invalid_request_error", message), status_code=400)

    def capability_fallbacks(body: dict) -> int:
        needs = request_capabilities(body)
        if "vision" not in needs or pool.with_capabilities({"vision"}):
            return 0
        count = 0
        for msg in body.get("messages") or []:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    extracted = ocr.extract_text_from_block(block)
                    if extracted:
                        text = f"[image OCR fallback]\n{extracted}"
                    else:
                        text = (
                            "[image block received; no vision backend is configured. "
                            "OCR/caption fallback could not extract text, so proceed from surrounding text.]"
                        )
                    block.clear()
                    block.update({
                        "type": "text",
                        "text": text,
                    })
                    count += 1
        return count

    @app.post("/v1/messages")
    async def messages(request: Request):
        try:
            body = await request.json()
        except Exception:
            return invalid_request("body is not valid JSON")
        if "messages" not in body or "max_tokens" not in body:
            return invalid_request("'messages' and 'max_tokens' are required")
        fallback_count = capability_fallbacks(body)
        try:
            conv = decode(body)
        except (KeyError, TypeError, AttributeError) as exc:
            return invalid_request(f"could not decode request: {exc!r}")
        skey = session_key(body)
        metrics: dict = {}
        metrics["client_tool_count"] = len(conv.tools)
        metrics["client_tool_names"] = [tool.name for tool in conv.tools]
        metrics.setdefault("tool_success_after_preflight", 0)
        metrics.setdefault("tool_failure_after_preflight", 0)
        metrics.setdefault("tool_results_after_preflight", [])
        for turn in conv.turns:
            for part in turn.parts:
                if not isinstance(part, ToolResultPart):
                    continue
                pending = pending_preflight.pop((skey, part.tool_call_id), None)
                if pending is None:
                    continue
                if part.is_error:
                    metrics["tool_failure_after_preflight"] += 1
                else:
                    metrics["tool_success_after_preflight"] += 1
                metrics["tool_results_after_preflight"].append({
                    "id": part.tool_call_id,
                    "tool": pending.get("tool"),
                    "preflight_reason": pending.get("reason"),
                    "success": not part.is_error,
                })

        _dump(settings, "anthropic-request", body)
        role = request_role(body, settings)
        if role == "reasoning":
            conv = _readonly_reasoning(conv, settings)
        chosen = router.pick(body)
        # The compaction budget depends on which backend serves the request,
        # so route first and pipeline against that backend's context window.
        req_settings = settings.model_copy(deep=True)
        req_settings.profile.context_window = chosen.cfg.context_window
        _apply_relaxed(req_settings, chosen.cfg.relaxed)
        conv = run_pipeline(conv, req_settings, stages, metrics)
        metrics["pipeline_tool_count"] = len(conv.tools)
        metrics["pipeline_tool_names"] = [tool.name for tool in conv.tools]
        metrics["pipeline_all_tool_count"] = len(conv.all_tools or conv.tools)
        rendered = chosen.profile.render(conv, chosen.model_name)
        apply_reasoning_budget(rendered, req_settings, chosen, role, body, conv, metrics)
        _dump(settings, "rendered-payload", rendered)

        stats["requests"] += 1
        msg_id = "msg_" + uuid.uuid4().hex[:24]
        model = body.get("model", chosen.model_name)
        metrics.update({"request_max_tokens": conv.params.max_tokens})
        metrics["memory_tokens"] = injected_memory_tokens(conv.system, counter)
        metrics["capability_fallbacks"] = fallback_count
        metrics["routing_intent"] = role
        metrics["routing_reason"] = "heuristic"
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
        if role in ("main", "reasoning"):
            if req_settings.research.enabled:
                try:
                    brief = await research.ensure(conv, pool, metrics)
                    fact = memory_fact(conv, brief or "")
                    if fact and settings.memory.enabled:
                        memory.merge(project_key(conv.system), fact)
                    conv = research.inject(conv, brief)
                    rendered = chosen.profile.render(conv, chosen.model_name)
                    apply_reasoning_budget(rendered, req_settings, chosen, role, body, conv, metrics)
                    _dump(settings, "rendered-payload", rendered)
                except Exception as exc:
                    metrics["research_error"] = str(exc)
        if role == "main":
            if req_settings.planning.enabled:
                metrics.setdefault("plan_drift", 0)
                try:
                    await planner.ensure(
                        skey,
                        conv,
                        pool,
                        metrics,
                        logger=logger,
                        parent_request_id=msg_id,
                        account_usage=account_usage,
                    )
                    conv = planner.inject(skey, conv)
                    rendered = chosen.profile.render(conv, chosen.model_name)
                    apply_reasoning_budget(rendered, req_settings, chosen, role, body, conv, metrics)
                    _dump(settings, "rendered-payload", rendered)
                except BackendError as exc:
                    metrics["plan_error"] = str(exc)
                    metrics.setdefault("plan_drift", 0)
            try:
                conv = await critic.maybe_inject(
                    skey,
                    conv,
                    pool,
                    metrics,
                    logger=logger,
                    parent_request_id=msg_id,
                    account_usage=account_usage,
                    record_critic=lambda r: _record_critic_stats(stats["critic"], r),
                )
                rendered = chosen.profile.render(conv, chosen.model_name)
                apply_reasoning_budget(rendered, req_settings, chosen, role, body, conv, metrics)
                _dump(settings, "rendered-payload", rendered)
            except BackendError as exc:
                metrics["critic_error"] = str(exc)

        cacheable = req_settings.cache.enabled and role in req_settings.cache.roles
        cache_key = payload_key(rendered) if cacheable else None
        cached_events = rcache.get(cache_key) if cache_key else None
        if cached_events is not None:
            record["cache"] = "response"
            events = _replay(cached_events)
        else:
            review_cb = None
            if role == "main" and req_settings.review.enabled:
                async def review_cb(trigger, review_conv, message, review_metrics):
                    return await reviewer.review(
                        trigger,
                        review_conv,
                        message,
                        pool,
                        review_metrics,
                        logger=logger,
                        parent_request_id=msg_id,
                        session_key=skey,
                        account_usage=account_usage,
                    )
            events = relay.run(
                conv,
                chosen.profile,
                chosen,
                req_settings,
                metrics=metrics,
                reviewer=review_cb,
                role=role,
                body=body,
            )

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
                    account_usage(chosen, ev, skey)
                elif isinstance(ev, ThinkingDelta):
                    metrics["reasoning_tokens_observed"] = (
                        metrics.get("reasoning_tokens_observed", 0)
                        + counter.count_text(ev.text)
                    )
                if (cache_key and cached_events is None) or settings.traces.enabled:
                    buffer.append(ev)
                yield ev

        def _finish_record() -> None:
            record["wall_ms"] = int((time.monotonic() - start) * 1000)
            if record["ttft_ms"] is None:
                record["ttft_ms"] = record["wall_ms"]
            record.update(metrics)
            for event in record.get("preflight_events") or []:
                if event.get("decision") != "rewrite" or not event.get("id"):
                    continue
                emitted = any(
                    call.get("id") == event.get("id")
                    for call in record.get("emitted_tool_calls") or []
                )
                if emitted:
                    pending_preflight[(skey, event["id"])] = event
            _record_runtime_stats(stats["runtime"], record)
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
            if kv_used is not None:
                b.kv_used, b.kv_used_ts = kv_used, time.monotonic()
            elif time.monotonic() - b.kv_used_ts < KV_USED_TTL_S:
                kv_used = b.kv_used  # hold last good reading across a missed poll
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
                "kv_used_pct": kv_used["pct"] if kv_used else None,
                "kv_used_est": kv_used["est"] if kv_used else False,
            }
        return JSONResponse({
            **stats,
            "backends": backends,
            "critic": _critic_summary(stats["critic"]),
            "runtime": _runtime_summary(stats["runtime"]),
            "response_cache": {"hits": rcache.hits, "misses": rcache.misses},
        })

    return app
