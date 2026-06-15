"""Session trace capture: the data flywheel.

Every request can be recorded as (rendered backend payload, IR events,
metrics) tagged with HARNESS_TRACE_TAG. The eval runner sets a unique tag
per trial and records the trial outcome; scripts/corpus.py joins the two
into an outcome-labeled SFT corpus.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from harness.ir import Done, IREvent, TextDelta, ThinkingDelta, ToolCall
from harness.rotation import DEFAULT_ROTATE_BYTES, rotate_if_needed


def serialize_event(ev: IREvent) -> dict:
    if isinstance(ev, TextDelta):
        return {"t": "text", "text": ev.text}
    if isinstance(ev, ThinkingDelta):
        return {"t": "thinking", "text": ev.text}
    if isinstance(ev, ToolCall):
        return {"t": "tool_call", "id": ev.id, "name": ev.name, "arguments": ev.arguments}
    if isinstance(ev, Done):
        return {"t": "done", "stop_reason": ev.stop_reason,
                "input_tokens": ev.input_tokens, "output_tokens": ev.output_tokens,
                "cached_tokens": ev.cached_tokens}
    raise TypeError(f"unknown event {ev!r}")


def assistant_message(events: list[dict]) -> dict:
    """Rebuild the OpenAI-format assistant message a trace's events represent."""
    text = "".join(e["text"] for e in events if e["t"] == "text")
    tool_calls = [
        {
            "id": e["id"],
            "type": "function",
            "function": {"name": e["name"], "arguments": json.dumps(e["arguments"])},
        }
        for e in events
        if e["t"] == "tool_call"
    ]
    msg: dict = {"role": "assistant", "content": text}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


class TraceStore:
    def __init__(
        self,
        directory: str | Path | None,
        tag: str | None = None,
        rotate_bytes: int = DEFAULT_ROTATE_BYTES,
    ) -> None:
        self.path = Path(directory) / "sessions.jsonl" if directory else None
        self.tag = tag if tag is not None else os.environ.get("HARNESS_TRACE_TAG", "")
        self.rotate_bytes = rotate_bytes
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            rotate_if_needed(self.path, self.rotate_bytes)

    def append(
        self,
        session_key: str,
        request_id: str,
        payload: dict,
        events: list[IREvent],
        metrics: dict,
    ) -> None:
        if not self.path:
            return
        line = json.dumps(
            {
                "ts": time.time(),
                "tag": self.tag,
                "session_key": session_key,
                "request_id": request_id,
                "payload": payload,
                "events": [serialize_event(e) for e in events],
                "metrics": metrics,
            },
            separators=(",", ":"),
        )
        with self._lock:
            rotate_if_needed(self.path, self.rotate_bytes)
            with self.path.open("a") as f:
                f.write(line + "\n")
