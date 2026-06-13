"""Generate harness TOML configs for eval runs: baseline, full, ablations."""

from __future__ import annotations

from pathlib import Path

STAGES = {
    "system_prompt": ('system_prompt = "replace"', 'system_prompt = "passthrough"'),
    "tool_prune": ("tool_prune = true", "tool_prune = false"),
    "tool_catalog": ("tool_catalog = true", "tool_catalog = false"),
    "fewshot": ("fewshot = true", "fewshot = false"),
    "repair": ("repair_retries = 2", "repair_retries = 0"),
}


def render(
    backend_url: str,
    model: str,
    profile: str,
    kind: str,
    port: int,
    log_path: str,
    overrides: dict[str, bool],
    traces_dir: str | None = None,
) -> str:
    lines = [
        "[server]",
        f"port = {port}",
        "",
        "[backend]",
        f'kind = "{kind}"',
        f'base_url = "{backend_url}"',
        f'model = "{model}"',
        "",
        "[profile]",
        f'name = "{profile}"',
        "",
        "[pipeline]",
    ]
    for stage, (on, off) in STAGES.items():
        lines.append(on if overrides.get(stage, True) else off)
    lines += ["", "[log]", f'requests_path = "{log_path}"']
    if traces_dir:
        lines += ["", "[traces]", "enabled = true", f'dir = "{traces_dir}"']
    return "\n".join(lines) + "\n"


def config_matrix() -> dict[str, dict[str, bool]]:
    """Named configs: baseline (all off), full (all on), one ablation per stage."""
    matrix = {
        "baseline": {s: False for s in STAGES},
        "full": {s: True for s in STAGES},
    }
    for stage in STAGES:
        matrix[f"ablate-{stage}"] = {s: (s != stage) for s in STAGES}
    return matrix


def write_configs(out_dir: Path, names: list[str], **kw) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = config_matrix()
    paths = {}
    for name in names:
        path = out_dir / f"{name}.toml"
        path.write_text(render(overrides=matrix[name], **kw))
        paths[name] = path
    return paths
