"""Encode IR stream events as Anthropic Messages API output (SSE or JSON).

Invariant: every event sequence we emit is spec-valid, including on error
paths — Claude Code must never see a half-open or malformed stream.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

from harness.ir import Done, IREvent, TextDelta, ThinkingDelta, ToolCall


def _ev(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def error_sse(error_type: str, message: str) -> str:
    return _ev("error", {"type": "error", "error": {"type": error_type, "message": message}})


def error_body(error_type: str, message: str) -> dict:
    return {"type": "error", "error": {"type": error_type, "message": message}}


class _BlockState:
    """Tracks the currently open content block and assigns indexes."""

    def __init__(self) -> None:
        self.index = -1
        self.open_kind: str | None = None  # "thinking" | "text" | None

    def open(self, kind: str, block: dict) -> str:
        self.index += 1
        self.open_kind = kind if kind in ("thinking", "text") else None
        return _ev(
            "content_block_start",
            {"type": "content_block_start", "index": self.index, "content_block": block},
        )

    def close(self) -> str:
        self.open_kind = None
        return _ev("content_block_stop", {"type": "content_block_stop", "index": self.index})

    def delta(self, delta: dict) -> str:
        return _ev(
            "content_block_delta",
            {"type": "content_block_delta", "index": self.index, "delta": delta},
        )


async def stream_sse(
    events: AsyncIterator[IREvent], model: str, msg_id: str
) -> AsyncIterator[str]:
    yield _ev(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )
    yield _ev("ping", {"type": "ping"})

    st = _BlockState()
    stop_reason = "end_turn"
    usage = {"input_tokens": 0, "output_tokens": 0}

    async for ev in events:
        if isinstance(ev, ThinkingDelta):
            if st.open_kind != "thinking":
                if st.open_kind:
                    yield st.close()
                yield st.open("thinking", {"type": "thinking", "thinking": ""})
            yield st.delta({"type": "thinking_delta", "thinking": ev.text})
        elif isinstance(ev, TextDelta):
            if st.open_kind != "text":
                if st.open_kind:
                    yield st.close()
                yield st.open("text", {"type": "text", "text": ""})
            yield st.delta({"type": "text_delta", "text": ev.text})
        elif isinstance(ev, ToolCall):
            if st.open_kind:
                yield st.close()
            yield st.open(
                "tool_use",
                {"type": "tool_use", "id": ev.id, "name": ev.name, "input": {}},
            )
            yield st.delta(
                {"type": "input_json_delta", "partial_json": json.dumps(ev.arguments)}
            )
            yield st.close()
        elif isinstance(ev, Done):
            stop_reason = ev.stop_reason
            usage = {"input_tokens": ev.input_tokens, "output_tokens": ev.output_tokens}

    if st.open_kind:
        yield st.close()
    yield _ev(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": usage["output_tokens"]},
        },
    )
    yield _ev("message_stop", {"type": "message_stop"})


def collect(events: list[IREvent], model: str, msg_id: str) -> dict:
    """Accumulate IR events into a non-streaming Anthropic message."""
    content: list[dict] = []
    stop_reason = "end_turn"
    usage = {"input_tokens": 0, "output_tokens": 0}

    def _append_text(kind: str, key: str, text: str) -> None:
        if content and content[-1]["type"] == kind:
            content[-1][key] += text
        else:
            block = {"type": kind, key: text}
            if kind == "thinking":
                block["signature"] = ""
            content.append(block)

    for ev in events:
        if isinstance(ev, ThinkingDelta):
            _append_text("thinking", "thinking", ev.text)
        elif isinstance(ev, TextDelta):
            _append_text("text", "text", ev.text)
        elif isinstance(ev, ToolCall):
            content.append(
                {"type": "tool_use", "id": ev.id, "name": ev.name, "input": ev.arguments}
            )
        elif isinstance(ev, Done):
            stop_reason = ev.stop_reason
            usage = {"input_tokens": ev.input_tokens, "output_tokens": ev.output_tokens}

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }
