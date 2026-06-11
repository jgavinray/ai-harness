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


class PipelineCfg(BaseModel):
    system_prompt: str = "replace"  # replace | compress | passthrough
    tool_prune: bool = True
    max_tools: int = 8
    fewshot: bool = True
    repair_retries: int = 2
    recent_turns_protected: int = 4
    reasoning: str = "thinking"  # thinking | strip


class DebugCfg(BaseModel):
    dump_prompts: bool = False
    dump_dir: str = "debug_dumps"


class Settings(BaseModel):
    server: ServerCfg = ServerCfg()
    backend: BackendCfg = BackendCfg()
    profile: ProfileCfg = ProfileCfg()
    pipeline: PipelineCfg = PipelineCfg()
    debug: DebugCfg = DebugCfg()


def load_settings(path: str | Path | None = None) -> Settings:
    if path and Path(path).exists():
        return Settings.model_validate(tomllib.loads(Path(path).read_text()))
    return Settings()
