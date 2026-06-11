"""Stage ③: simplify tool schemas for small models.

Trims verbose descriptions, strips JSON-schema noise keys, and flattens
nullable anyOf unions. original_schema is left untouched — the repair
stage validates against it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from harness.config import Settings
from harness.ir import Conversation, ToolDef

NOISE_KEYS = ("$schema", "title", "additionalProperties")
TOOL_DESC_MAX = 300
PROP_DESC_MAX = 150


def trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = cut.rfind(". ")
    if boundary > 0:
        return cut[: boundary + 1]
    return cut[: limit - 1].rstrip() + "…"


def _simplify(schema: Any) -> Any:
    if isinstance(schema, list):
        return [_simplify(s) for s in schema]
    if not isinstance(schema, dict):
        return schema

    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key in NOISE_KEYS:
            continue
        if key == "description" and isinstance(value, str):
            out[key] = trim(value, PROP_DESC_MAX)
        else:
            out[key] = _simplify(value)

    any_of = out.get("anyOf")
    if isinstance(any_of, list):
        non_null = [s for s in any_of if s != {"type": "null"}]
        if len(non_null) == 1:
            del out["anyOf"]
            out.update(non_null[0])
    return out


class ToolSchemaStage:
    def apply(self, conv: Conversation, settings: Settings) -> Conversation:
        tools = tuple(
            ToolDef(
                t.name,
                trim(t.description, TOOL_DESC_MAX),
                _simplify(t.input_schema),
                t.original_schema,
            )
            for t in conv.tools
        )
        return replace(conv, tools=tools)
