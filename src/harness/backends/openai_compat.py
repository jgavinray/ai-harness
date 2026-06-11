"""OpenAI Chat Completions streaming client — the universal downstream
protocol (Ollama, vLLM, llama.cpp server, LM Studio, OpenRouter, ...).

vLLM and llama.cpp subclasses add their schema-constraint parameters,
used by the relay on repair retries.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from harness.backends.base import Backend, BackendError
from harness.config import BackendCfg


class OpenAIBackend(Backend):
    async def stream(self, payload: dict[str, Any]) -> AsyncIterator[dict]:
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        try:
            async with self.client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode(errors="replace")[:500]
                    raise BackendError(f"backend HTTP {resp.status_code}: {body}")
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        return
                    yield json.loads(data)
        except (httpx.HTTPError, RuntimeError) as exc:
            raise BackendError(f"backend stream failed: {exc}") from exc


class VllmBackend(OpenAIBackend):
    constrained = True

    def apply_constraint(self, payload: dict[str, Any], schema: dict) -> dict[str, Any]:
        payload["guided_json"] = schema
        payload["tool_choice"] = "required"
        return payload


class LlamaCppBackend(OpenAIBackend):
    constrained = True

    def stream(self, payload: dict[str, Any]) -> AsyncIterator[dict]:
        payload.setdefault("cache_prompt", True)  # llama.cpp KV prefix reuse
        return super().stream(payload)

    def apply_constraint(self, payload: dict[str, Any], schema: dict) -> dict[str, Any]:
        payload["json_schema"] = schema
        return payload


KINDS = {"openai": OpenAIBackend, "vllm": VllmBackend, "llamacpp": LlamaCppBackend}


def make_backend(cfg: BackendCfg, client: httpx.AsyncClient) -> Backend:
    try:
        return KINDS[cfg.kind](cfg, client)
    except KeyError:
        raise ValueError(f"unknown backend kind {cfg.kind!r}; available: {sorted(KINDS)}") from None
