import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))
import configs as eval_configs  # noqa: E402
import report as eval_report  # noqa: E402

TASKS = Path("evals/tasks")


def test_config_matrix():
    m = eval_configs.config_matrix()
    assert set(m["baseline"].values()) == {False}
    assert set(m["full"].values()) == {True}
    assert m["ablate-fewshot"]["fewshot"] is False
    assert m["ablate-fewshot"]["tool_prune"] is True


def test_render_baseline_passthrough(tmp_path):
    paths = eval_configs.write_configs(
        tmp_path, ["baseline", "full"],
        backend_url="http://b/v1", model="m", profile="qwen", kind="vllm",
        port=9999, log_path="/tmp/l.jsonl",
    )
    baseline = paths["baseline"].read_text()
    assert 'system_prompt = "passthrough"' in baseline
    assert "repair_retries = 0" in baseline
    full = paths["full"].read_text()
    assert 'system_prompt = "replace"' in full
    # generated configs must be loadable by the harness
    from harness.config import load_settings
    s = load_settings(paths["baseline"])
    assert s.server.port == 9999 and s.backend.kind == "vllm"


def test_report_aggregation(tmp_path):
    rows = [
        {"model": "m", "config": "full", "success": True, "timed_out": False,
         "valid_calls": 8, "invalid_calls": 0, "repaired_calls": 1, "retries": 1,
         "input_tokens": 1000, "output_tokens": 100, "session_wall_s": 30},
        {"model": "m", "config": "full", "success": False, "timed_out": True,
         "valid_calls": 2, "invalid_calls": 2, "repaired_calls": 0, "retries": 2,
         "input_tokens": 500, "output_tokens": 50, "session_wall_s": 300},
    ]
    p = tmp_path / "results.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    agg = eval_report.aggregate(eval_report.load(p))
    m = agg[("m", "full")]
    assert m["success_rate"] == 0.5
    assert m["timeout_rate"] == 0.5
    assert m["post_repair_invalid_rate"] == 2 / 12
    assert m["retries_per_session"] == 1.5
    md = eval_report.markdown(agg)
    assert "| m | full |" in md


def test_all_tasks_complete_and_checkers_valid():
    names = {p.name for p in TASKS.iterdir()}
    assert names == {"fix-test", "add-endpoint", "rename-refactor", "find-and-report", "multi-step"}
    for task in TASKS.iterdir():
        assert (task / "prompt.txt").exists(), task
        assert (task / "check.sh").exists(), task
        assert (task / "repo_template").is_dir(), task
        subprocess.run(["bash", "-n", str(task / "check.sh")], check=True)


def test_broken_tasks_fail_their_own_checks(tmp_path):
    """Initial repo state must FAIL the checker (otherwise the task tests nothing)."""
    import shutil
    for name in ("fix-test", "add-endpoint", "multi-step", "rename-refactor"):
        work = tmp_path / name
        shutil.copytree(TASKS / name / "repo_template", work)
        shutil.copy(TASKS / name / "check.sh", work / "check.sh")
        subprocess.run(["git", "init", "-q"], cwd=work, check=True)
        r = subprocess.run(["bash", "check.sh"], cwd=work, capture_output=True)
        assert r.returncode != 0, f"{name} passes its checker without any work"
