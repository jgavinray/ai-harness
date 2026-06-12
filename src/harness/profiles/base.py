"""Model profiles: render IR → OpenAI-style payload, parse stream → IR events.

A profile owns every model-family quirk: system-role support, reasoning-tag
handling, and (via overrides) anything a family needs differently.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Iterator

from harness.ir import (
    Conversation,
    Done,
    IREvent,
    TextDelta,
    TextPart,
    ThinkingDelta,
    ThinkingPart,
    ToolCall,
    ToolCallPart,
    ToolResultPart,
)

FINISH_MAP = {"tool_calls": "tool_use", "length": "max_tokens"}


class TagSplitter:
    """Routes streamed text inside open/close tags to a 'think' channel.

    Handles tags split across chunk boundaries by holding back up to
    len(tag)-1 trailing chars that could be a tag prefix.
    """

    def __init__(self, open_tag: str, close_tag: str) -> None:
        self.open_tag, self.close_tag = open_tag, close_tag
        self.buf = ""
        self.in_think = False

    def _current_tag(self) -> str:
        return self.close_tag if self.in_think else self.open_tag

    def feed(self, text: str) -> Iterator[tuple[str, str]]:
        self.buf += text
        while True:
            tag = self._current_tag()
            idx = self.buf.find(tag)
            if idx != -1:
                if idx:
                    yield ("think" if self.in_think else "text", self.buf[:idx])
                self.buf = self.buf[idx + len(tag):]
                self.in_think = not self.in_think
                continue
            # emit everything that cannot be the start of a partial tag
            holdback = 0
            for k in range(min(len(tag) - 1, len(self.buf)), 0, -1):
                if tag.startswith(self.buf[-k:]):
                    holdback = k
                    break
            emit, self.buf = self.buf[: len(self.buf) - holdback], self.buf[len(self.buf) - holdback:]
            if emit:
                yield ("think" if self.in_think else "text", emit)
            return

    def flush(self) -> Iterator[tuple[str, str]]:
        if self.buf:
            yield ("think" if self.in_think else "text", self.buf)
            self.buf = ""


class Profile:
    name = "base"
    supports_system_role = True
    reasoning_tags: tuple[str, str] | None = None

    # ---------- render ----------

    def render(self, conv: Conversation, model: str) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if conv.system and self.supports_system_role:
            messages.append({"role": "system", "content": conv.system})

        pending_system = conv.system if (conv.system and not self.supports_system_role) else None

        for turn in conv.turns:
            if turn.role == "assistant":
                text = "".join(
                    p.text for p in turn.parts if isinstance(p, TextPart)
                )
                tool_calls = [
                    {
                        "id": p.id,
                        "type": "function",
                        "function": {"name": p.name, "arguments": json.dumps(p.arguments)},
                    }
                    for p in turn.parts
                    if isinstance(p, ToolCallPart)
                ]
                msg: dict[str, Any] = {"role": "assistant", "content": text}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                messages.append(msg)
                continue

            # user turn: tool results become role=tool messages; text stays user
            texts = []
            for part in turn.parts:
                if isinstance(part, ToolResultPart):
                    content = part.content
                    if part.is_error:
                        content = f"ERROR: {content}"
                    messages.append(
                        {"role": "tool", "tool_call_id": part.tool_call_id, "content": content}
                    )
                elif isinstance(part, (TextPart, ThinkingPart)):
                    texts.append(part.text)
            if texts:
                content = "\n".join(texts)
                if pending_system is not None:
                    content = f"{pending_system}\n\n{content}"
                    pending_system = None
                messages.append({"role": "user", "content": content})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": conv.params.max_tokens,
            # Always stream from the backend regardless of what the client
            # asked for: the SSE parser is the only response path, and plain
            # JSON replies would yield no events and drop usage entirely.
            # The server collects events for non-streaming clients.
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if conv.params.temperature is not None:
            payload["temperature"] = conv.params.temperature
        if conv.params.stop_sequences:
            payload["stop"] = list(conv.params.stop_sequences)
        if conv.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in conv.tools
            ]
        return payload

    # ---------- parse ----------

    async def parse(self, chunks: AsyncIterator[dict]) -> AsyncIterator[IREvent]:
        splitter = (
            TagSplitter(*self.reasoning_tags) if self.reasoning_tags else None
        )
        calls: dict[int, dict[str, str]] = {}
        finish: str | None = None
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        cached = 0
        evaluated: int | None = None  # llama.cpp timings.prompt_n

        async for chunk in chunks:
            if chunk.get("usage"):
                usage.update({k: v for k, v in chunk["usage"].items() if v})
                details = chunk["usage"].get("prompt_tokens_details") or {}
                cached = max(cached, details.get("cached_tokens") or 0)
            if chunk.get("timings"):
                evaluated = chunk["timings"].get("prompt_n", evaluated)
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content"):
                    yield ThinkingDelta(delta["reasoning_content"])
                if delta.get("content"):
                    if splitter:
                        for kind, text in splitter.feed(delta["content"]):
                            yield ThinkingDelta(text) if kind == "think" else TextDelta(text)
                    else:
                        yield TextDelta(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    slot = calls.setdefault(tc["index"], {"id": "", "name": "", "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]

        if splitter:
            for kind, text in splitter.flush():
                yield ThinkingDelta(text) if kind == "think" else TextDelta(text)

        for slot in calls.values():
            raw = slot["arguments"]
            try:
                args = json.loads(raw) if raw else {}
                yield ToolCall(slot["id"], slot["name"], args)
            except json.JSONDecodeError:
                yield ToolCall(slot["id"], slot["name"], {}, raw_arguments=raw)

        stop_reason = FINISH_MAP.get(finish or "stop", "end_turn")
        if calls and stop_reason == "end_turn":
            stop_reason = "tool_use"
        if not cached and evaluated is not None and usage["prompt_tokens"]:
            cached = max(usage["prompt_tokens"] - evaluated, 0)
        yield Done(
            stop_reason, usage["prompt_tokens"], usage["completion_tokens"], cached
        )
