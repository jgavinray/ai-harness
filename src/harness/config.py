import tomllib
from pathlib import Path
from typing import Literal

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
    relaxed: list[str] = []  # eval-gated scaffolds this backend no longer needs
    # Concurrency the *hardware* sustains, not what the engine accepts;
    # None = unlimited. A backend at this limit is skipped during routing.
    max_in_flight: int | None = None


class PipelineCfg(BaseModel):
    policy_owner: Literal["harness", "agentic_os"] = "harness"
    system_prompt: str = "replace"  # replace | compress | passthrough
    tool_prune: bool = True
    tool_catalog: bool = True  # list the full tool inventory in the system prompt
    # (only active when tool_prune is on; unpruned requests surface every schema)
    max_tools: int = 8
    fewshot: bool = True
    repair_retries: int = 2
    recent_turns_protected: int = 4
    effective_context_window: int | None = None
    compact_at_ratio: float = 0.80
    compact_target_ratio: float = 0.50
    action_state_tools: bool = True
    reasoning: str = "thinking"  # thinking | strip
    workflow_guards: bool = True
    guard_edit_without_read: bool = True
    guard_verify_after_edit: bool = True
    allowed_roots: list[str] = []


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


class RoutingCfg(BaseModel):
    reasoning: bool = True
    reasoning_readonly_tools: list[str] = ["Read", "Grep", "Glob", "LS", "WebFetch"]


class ReviewCfg(BaseModel):
    enabled: bool = False
    max_chars: int = 6000
    max_tokens: int = 500
    triggers: list[str] = [
        "plan_drift",
        "verify_after_edit",
        "loop_break",
        "invalid_tool_retry",
    ]


class CriticCfg(BaseModel):
    enabled: bool = False
    max_chars: int = 12000
    max_tokens: int = 1200
    min_tool_calls: int = 1
    triggers: list[str] = [
        "risky_path",
        "edit",
        "build_failure",
        "test_failure",
        "tool_error",
    ]


class ReasoningBudgetCfg(BaseModel):
    enabled: bool = False
    default_tokens: int = 1024
    max_auto_tokens: int = 8192
    max_manual_tokens: int = 32768
    final_answer_reserve: int = 4096
    load_shed: bool = True
    fallback_when_unavailable: str = "degrade"  # degrade | skip | fail
    role_tokens: dict[str, int] = {
        "reasoning": 2048,
        "plan": 4096,
        "review": 2048,
        "critic": 4096,
    }
    mode_tokens: dict[str, int] = {
        "file_edit": 1024,
        "hard_file_edit": 4096,
        "project_survey": 8192,
        "architecture_plan": 16384,
        "deep_architecture_plan": 32768,
        "integration_debug": 8192,
        "kernel_change_plan": 16384,
        "kernel_critic": 8192,
    }


class RiskProfileCfg(BaseModel):
    name: str
    path_patterns: list[str] = []
    text_patterns: list[str] = []
    plan_mode: str | None = None
    critic_mode: str | None = None


class SkillsCfg(BaseModel):
    enabled: bool = False
    dir: str = "~/.codex/skills"
    cache_dir: str = "~/.ai-harness/compiled-skills"
    max_tokens: int = 400


class ResearchCfg(BaseModel):
    enabled: bool = False
    cache_dir: str = "~/.ai-harness/research"
    max_chars: int = 12000
    chunk_chars: int = 4000


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
    routing: RoutingCfg = RoutingCfg()
    review: ReviewCfg = ReviewCfg()
    critic: CriticCfg = CriticCfg()
    reasoning_budget: ReasoningBudgetCfg = ReasoningBudgetCfg()
    risk_profiles: list[RiskProfileCfg] = []
    skills: SkillsCfg = SkillsCfg()
    research: ResearchCfg = ResearchCfg()


def load_settings(path: str | Path | None = None) -> Settings:
    if path and Path(path).exists():
        return Settings.model_validate(tomllib.loads(Path(path).read_text()))
    return Settings()
