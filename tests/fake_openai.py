"""Scripted OpenAI-compatible fake for tests.

Each entry in FakeOpenAI.scripts is consumed per request (last one repeats).
A script is a list of chunk dicts streamed as SSE, or special entries:
  {"_status": 500}        → HTTP error response
  {"_die_midstream": True} → connection drops after prior chunks
Requests received are recorded in .requests for assertions.
"""

from __future__ import annotations

import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse


class FakeOpenAI:
    def __init__(self) -> None:
        self.app = FastAPI()
        self.scripts: list[list[dict]] = []
        self.requests: list[dict] = []
        self.metrics_text: str | None = None  # set to expose a /metrics endpoint
        self.slots: list[dict] | None = None  # set to expose a /slots endpoint
        self.app.post("/v1/chat/completions")(self.handler)
        self.app.get("/metrics")(self.metrics)
        self.app.get("/slots")(self.get_slots)

    async def metrics(self):
        if self.metrics_text is None:
            # mirrors llama.cpp without --metrics
            return PlainTextResponse("not enabled", status_code=501)
        return PlainTextResponse(self.metrics_text)

    async def get_slots(self):
        if self.slots is None:
            return JSONResponse({"error": "slots disabled"}, status_code=501)
        return JSONResponse(self.slots)

    def push(self, script: list[dict]) -> None:
        self.scripts.append(script)

    async def handler(self, request: Request):
        body = await request.json()
        self.requests.append(body)
        script = self.scripts.pop(0) if len(self.scripts) > 1 else self.scripts[0]
        if script and script[0].get("_status"):
            return JSONResponse({"error": "scripted failure"}, status_code=script[0]["_status"])

        if not body.get("stream"):
            # real servers answer stream=false with one JSON completion, not SSE
            text, finish, usage = "", "stop", {}
            for chunk in script:
                for choice in chunk.get("choices", []):
                    text += (choice.get("delta") or {}).get("content") or ""
                    finish = choice.get("finish_reason") or finish
                usage = chunk.get("usage") or usage
            return JSONResponse({
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish,
                }],
                "usage": usage,
            })

        def gen():
            for chunk in script:
                if chunk.get("_die_midstream"):
                    raise RuntimeError("scripted mid-stream death")
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")


def text_chunk(text: str) -> dict:
    return {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}


def tool_chunk(call_id: str, name: str, arguments: str) -> dict:
    return {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {"index": 0, "id": call_id, "function": {"name": name, "arguments": arguments}}
                    ]
                },
                "finish_reason": None,
            }
        ]
    }


def finish_chunk(reason: str = "stop", prompt_tokens: int = 10, completion_tokens: int = 5) -> dict:
    return {
        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }
