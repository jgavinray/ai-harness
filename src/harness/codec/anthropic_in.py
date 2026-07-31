"""Decode an Anthropic Messages API request body into the IR."""

from __future__ import annotations

from typing import Any

from harness.ir import (
    Conversation,
    GenParams,
    Part,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolDef,
    ToolResultPart,
    Turn,
)

UNSUPPORTED = TextPart("[unsupported content omitted]")


def _flatten_system(system: str | list[dict[str, Any]] | None) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    return "\n\n".join(b.get("text", "") for b in system if b.get("type") == "text")


def _flatten_result_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    texts = []
    for block in content:
        if block.get("type") == "text":
            texts.append(block.get("text", ""))
        else:
            texts.append(UNSUPPORTED.text)
    return "\n".join(texts)


def _decode_block(block: dict[str, Any]) -> Part:
    kind = block.get("type")
    if kind == "text":
        return TextPart(block.get("text", ""))
    if kind == "thinking":
        return ThinkingPart(block.get("thinking", ""))
    if kind == "tool_use":
        return ToolCallPart(block["id"], block["name"], block.get("input") or {})
    if kind == "tool_result":
        return ToolResultPart(
            block["tool_use_id"],
            _flatten_result_content(block.get("content")),
            bool(block.get("is_error", False)),
        )
    return UNSUPPORTED


def _response_schema(body: dict[str, Any]) -> dict[str, Any] | None:
    """The JSON schema from output_config.format, when the client asked for one.

    Anthropic's structured-output field. Any other format type (plain text) or
    a missing schema means the reply is unconstrained.
    """
    fmt = (body.get("output_config") or {}).get("format") or {}
    if fmt.get("type") != "json_schema":
        return None
    schema = fmt.get("schema")
    return schema if isinstance(schema, dict) else None


def decode(body: dict[str, Any]) -> Conversation:
    turns = []
    for msg in body.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            parts: tuple[Part, ...] = (TextPart(content),)
        else:
            parts = tuple(_decode_block(b) for b in content)
        turns.append(Turn(msg["role"], parts))

    # A tool definition without "input_schema" is a server-side tool (e.g.
    # {"type": "web_search_20250305", "name": "web_search"}): Anthropic's API
    # executes it, so a local backend can neither be given its schema nor run
    # it. Drop it — offering it would invite calls nothing can answer, and
    # failing the request takes the whole turn down with it.
    tools = tuple(
        ToolDef(
            t["name"],
            t.get("description", ""),
            t["input_schema"],
            t["input_schema"],
        )
        for t in body.get("tools", [])
        if isinstance(t.get("input_schema"), dict)
    )

    params = GenParams(
        max_tokens=body.get("max_tokens", 4096),
        temperature=body.get("temperature"),
        stop_sequences=tuple(body.get("stop_sequences") or ()),
        stream=bool(body.get("stream", False)),
        response_schema=_response_schema(body),
    )
    return Conversation(_flatten_system(body.get("system")), tuple(turns), tools, params)
