"""FastAPI app: the Anthropic Messages API surface Claude Code talks to.

Every response path — success, backend failure, bad request — produces a
spec-valid Anthropic response so Claude Code's own retry/UX logic works.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from harness import relay
from harness.backends.base import BackendError
from harness.backends.openai_compat import make_backend
from harness.codec.anthropic_in import decode
from harness.codec.anthropic_out import collect, error_body, error_sse, stream_sse
from harness.config import Settings
from harness.pipeline.base import run_pipeline
from harness.pipeline.fewshot import FewshotStage
from harness.pipeline.history import HistoryStage
from harness.pipeline.system_prompt import SystemPromptStage
from harness.pipeline.tool_prune import ToolPruneStage
from harness.pipeline.tool_schema import ToolSchemaStage
from harness.profiles.registry import get_profile
from harness.tokens.counter import HeuristicCounter, count_conversation

STAGES = [
    SystemPromptStage(),
    ToolPruneStage(),
    ToolSchemaStage(),
    HistoryStage(),
    FewshotStage(),
]


def _dump(settings: Settings, kind: str, data: dict) -> None:
    if not settings.debug.dump_prompts:
        return
    d = Path(settings.debug.dump_dir)
    d.mkdir(parents=True, exist_ok=True)
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}-{kind}.json"
    (d / name).write_text(json.dumps(data, indent=2))


def create_app(settings: Settings, backend_client: httpx.AsyncClient | None = None) -> FastAPI:
    app = FastAPI(title="ai-harness")
    client = backend_client or httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10))
    backend = make_backend(settings.backend, client)
    profile = get_profile(settings.profile.name)
    counter = HeuristicCounter()
    stats = {"requests": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0}

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
        conv = run_pipeline(conv, settings, STAGES)
        _dump(settings, "rendered-payload", profile.render(conv, settings.backend.model))

        stats["requests"] += 1
        msg_id = "msg_" + uuid.uuid4().hex[:24]
        model = body.get("model", settings.backend.model)
        events = relay.run(conv, profile, backend, settings)

        if conv.params.stream:
            async def sse():
                try:
                    async for piece in stream_sse(_track(events), model, msg_id):
                        yield piece
                except BackendError as exc:
                    stats["errors"] += 1
                    yield error_sse("overloaded_error", str(exc))

            return StreamingResponse(sse(), media_type="text/event-stream")

        try:
            collected = [e async for e in _track(events)]
        except BackendError as exc:
            stats["errors"] += 1
            return JSONResponse(error_body("overloaded_error", str(exc)), status_code=529)
        return JSONResponse(collect(collected, model, msg_id))

    async def _track(events):
        async for ev in events:
            if hasattr(ev, "input_tokens"):
                stats["input_tokens"] += ev.input_tokens
                stats["output_tokens"] += ev.output_tokens
            yield ev

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        try:
            body = await request.json()
            conv = decode(body)
        except Exception as exc:
            return invalid_request(f"could not decode request: {exc!r}")
        return JSONResponse({"input_tokens": count_conversation(conv, counter)})

    @app.get("/stats")
    async def get_stats():
        return JSONResponse(stats)

    return app
