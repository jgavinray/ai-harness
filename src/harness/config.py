import tomllib
from pathlib import Path

from pydantic import BaseModel


class ServerCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8484


class BackendCfg(BaseModel):
    kind: str = "openai"  # openai | vllm | llamacpp
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5-coder:14b"
    api_key: str = "local"


class ProfileCfg(BaseModel):
    name: str = "qwen"  # qwen | deepseek_r1 | devstral | gemma
    context_window: int = 32768


class PoolBackendCfg(BackendCfg):
    """One entry of the [[backends]] fleet array."""

    name: str
    profile: str = "qwen"
    context_window: int = 32768
    roles: list[str] = ["main", "subagent", "fast"]
    capabilities: list[str] = []
    # Concurrency the *hardware* sustains, not what the engine accepts;
    # None = unlimited. A backend at this limit is skipped during routing.
    max_in_flight: int | None = None


class PipelineCfg(BaseModel):
    system_prompt: str = "replace"  # replace | compress | passthrough
    tool_prune: bool = True
    tool_catalog: bool = True  # list the full tool inventory in the system prompt
    # (only active when tool_prune is on; unpruned requests surface every schema)
    max_tools: int = 8
    fewshot: bool = True
    repair_retries: int = 2
    recent_turns_protected: int = 4
    reasoning: str = "thinking"  # thinking | strip
    workflow_guards: bool = True
    guard_edit_without_read: bool = True
    guard_verify_after_edit: bool = True


class DebugCfg(BaseModel):
    dump_prompts: bool = False
    dump_dir: str = "debug_dumps"


class LogCfg(BaseModel):
    requests_path: str | None = None  # JSONL per-request log; None = disabled


class TracesCfg(BaseModel):
    enabled: bool = False
    dir: str = "traces"


class MemoryCfg(BaseModel):
    enabled: bool = False
    dir: str = "~/.ai-harness/memory"
    idle_s: float = 300.0  # session considered finished after this much quiet
    max_chars: int = 4000  # ~1k tokens of injected memory


class PlanningCfg(BaseModel):
    enabled: bool = False
    max_steps: int = 8
    max_chars: int = 4000


class SkillsCfg(BaseModel):
    enabled: bool = False
    dir: str = "~/.codex/skills"
    cache_dir: str = "~/.ai-harness/compiled-skills"
    max_tokens: int = 400


class CacheCfg(BaseModel):
    enabled: bool = True
    ttl_s: float = 600.0
    max_entries: int = 256
    roles: list[str] = ["fast"]  # which routed roles are response-cacheable


class Settings(BaseModel):
    server: ServerCfg = ServerCfg()
    backend: BackendCfg = BackendCfg()
    backends: list[PoolBackendCfg] = []  # fleet mode; empty = single-backend mode
    profile: ProfileCfg = ProfileCfg()
    pipeline: PipelineCfg = PipelineCfg()
    debug: DebugCfg = DebugCfg()
    log: LogCfg = LogCfg()
    cache: CacheCfg = CacheCfg()
    traces: TracesCfg = TracesCfg()
    memory: MemoryCfg = MemoryCfg()
    planning: PlanningCfg = PlanningCfg()
    skills: SkillsCfg = SkillsCfg()


def load_settings(path: str | Path | None = None) -> Settings:
    if path and Path(path).exists():
        return Settings.model_validate(tomllib.loads(Path(path).read_text()))
    return Settings()
