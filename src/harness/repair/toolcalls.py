"""Validate and repair model tool calls against the ORIGINAL Anthropic schema.

Returns (repaired_call, None) on success or (None, error_message) when the
call cannot be made valid locally; the relay then retries with feedback.
"""

from __future__ import annotations

import json_repair
import jsonschema

from harness.ir import ToolCall, ToolDef


def _coerce_types(args: dict, schema: dict) -> dict:
    """Fix the common small-model quirk of JSON scalars arriving as strings
    ("false", "25") when the schema expects boolean/integer/number."""
    props = schema.get("properties")
    if not isinstance(props, dict):
        return args
    out = dict(args)
    for key, spec in props.items():
        if key not in out or not isinstance(spec, dict) or not isinstance(out[key], str):
            continue
        expected = spec.get("type")
        value = out[key].strip()
        if expected == "boolean" and value.lower() in ("true", "false"):
            out[key] = value.lower() == "true"
        elif expected == "integer":
            try:
                out[key] = int(value)
            except ValueError:
                pass
        elif expected == "number":
            try:
                out[key] = float(value) if "." in value else int(value)
            except ValueError:
                pass
        elif expected == "string" and key.endswith("path") and (
                # Small models wrap paths in literal quotes and the call still
                # validates as a string, so the client fails instead (live
                # 2026-07-11: file_path='"/home/.../CMakeLists.txt"', four Read
                # failures in a row). Only path-like keys: content-bearing
                # strings (old_string/new_string) can be legitimately quoted.
                len(value) > 2
                and value[0] == value[-1]
                and value[0] in ('"', "'")
                and value[0] not in value[1:-1]
            ):
                out[key] = value[1:-1]
    return out


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
    if isinstance(args, dict):
        args = _coerce_types(args, tool.original_schema)

    try:
        jsonschema.validate(args, tool.original_schema)
    except jsonschema.ValidationError as exc:
        return None, f"validation error for tool {call.name}: {exc.message}"
    return ToolCall(call.id, call.name, args), None
