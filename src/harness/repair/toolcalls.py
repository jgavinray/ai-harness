"""Validate and repair model tool calls against the ORIGINAL Anthropic schema.

Returns (repaired_call, None) on success or (None, error_message) when the
call cannot be made valid locally; the relay then retries with feedback.
"""

from __future__ import annotations

import json

import json_repair
import jsonschema

from harness.ir import ToolCall, ToolDef


def repair_toolcall(
    call: ToolCall, tools: tuple[ToolDef, ...]
) -> tuple[ToolCall | None, str | None]:
    tool = next((t for t in tools if t.name == call.name), None)
    if tool is None:
        names = ", ".join(t.name for t in tools)
        return None, f"unknown tool {call.name!r}; available tools: {names}"

    args = call.arguments
    if not args and call.raw_arguments:
        repaired = json_repair.loads(call.raw_arguments)
        if not isinstance(repaired, dict) or not repaired:
            return None, f"arguments are not a JSON object: {call.raw_arguments[:200]!r}"
        args = repaired

    try:
        jsonschema.validate(args, tool.original_schema)
    except jsonschema.ValidationError as exc:
        return None, f"validation error for tool {call.name}: {exc.message}"
    return ToolCall(call.id, call.name, args), None
