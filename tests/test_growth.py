import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import lora_train  # noqa: E402
import promote_candidate  # noqa: E402
import relax_scaffold  # noqa: E402
import shadow_eval  # noqa: E402


def test_shadow_eval_lists_candidate_commands(tmp_path):
    cfg = tmp_path / "harness.toml"
    cfg.write_text(
        '[[backends]]\nname = "live"\nbase_url = "http://live/v1"\nmodel = "m"\nroles = ["main"]\n'
        '[[backends]]\nname = "cand"\nbase_url = "http://cand/v1"\nmodel = "c"\n'
        'profile = "qwen"\nkind = "vllm"\nroles = ["candidate"]\n'
    )
    cmds = shadow_eval.candidate_commands(cfg, "out")
    assert len(cmds) == 1
    assert "--model c" in cmds[0]
    assert "--backend-url http://cand/v1" in cmds[0]


def test_promote_candidate_gate_and_config_edit(tmp_path):
    results = tmp_path / "results.jsonl"
    rows = [
        {"model": "inc", "success": True},
        {"model": "inc", "success": False},
        {"model": "cand", "success": True},
        {"model": "cand", "success": True},
    ]
    results.write_text("\n".join(json.dumps(r) for r in rows))
    assert promote_candidate.should_promote(results, "inc", "cand", 0.25)
    cfg = tmp_path / "harness.toml"
    cfg.write_text(
        '[[backends]]\nname = "cand"\nbase_url = "http://cand/v1"\nmodel = "c"\nroles = ["candidate"]\n'
    )
    promote_candidate.promote_config(cfg, "cand", ["main", "subagent"])
    assert 'roles = ["main", "subagent"]' in cfg.read_text()


def test_lora_train_command():
    cmd = lora_train.command("corpus.jsonl", "base-model", "adapters/out")
    assert cmd == [
        "mlx_lm.lora", "--model", "base-model", "--train",
        "--data", "corpus.jsonl", "--adapter-path", "adapters/out",
    ]


def test_lora_train_cuda_command():
    # Phase 2 (flywheel spec): QLoRA on the RTX Pro 6000 via a bundled
    # trainer script; mlx stays the Apple-silicon path.
    cmd = lora_train.command("c.jsonl", "base", "adapters/out", backend="cuda")
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("scripts/qlora_train.py")
    assert cmd[2:] == ["--model", "base", "--data", "c.jsonl", "--out", "adapters/out"]


def test_qlora_train_dry_run_needs_no_gpu_deps(tmp_path):
    # The trainer must resolve and print its config without importing
    # torch/transformers, so the threshold job can emit it on any host.
    import subprocess
    data = tmp_path / "c.jsonl"
    data.write_text(json.dumps({"messages": [{"role": "user", "content": "x"}]}) + "\n")
    r = subprocess.run(
        [sys.executable, str(Path("scripts/qlora_train.py").resolve()),
         "--model", "base", "--data", str(data), "--out", str(tmp_path / "a"),
         "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    cfg = json.loads(r.stdout)
    assert cfg["model"] == "base"
    assert cfg["rows"] == 1
    assert cfg["quant"] == "nf4"


def test_qlora_train_supports_unquantized_lora(tmp_path):
    # bitsandbytes has no CUDA aarch64 binary for the DGX Spark (GB10), so
    # the trainer must support plain bf16 LoRA (--quant none); the Spark's
    # 121 GB unified memory holds the 27B in bf16.
    import subprocess
    data = tmp_path / "c.jsonl"
    data.write_text(json.dumps({"messages": []}) + "\n")
    r = subprocess.run(
        [sys.executable, str(Path("scripts/qlora_train.py").resolve()),
         "--model", "base", "--data", str(data), "--out", str(tmp_path / "a"),
         "--quant", "none", "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    assert json.loads(r.stdout)["quant"] == "none"


def test_qlora_normalize_parses_tool_call_arguments():
    # Training failure 2026-07-09 on the Spark: the corpus keeps
    # tool_calls[].function.arguments as a JSON string (OpenAI wire format),
    # but Qwen's chat template iterates arguments as a mapping
    # (jinja2: "Can only get item pairs from a mapping").
    import qlora_train
    row = {"messages": [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "type": "function",
             "function": {"name": "Read", "arguments": "{\"file_path\": \"/x\"}"}},
            {"id": "t2", "type": "function",
             "function": {"name": "Bash", "arguments": "not json"}},
        ]},
        {"role": "tool", "content": "ok"},
    ]}
    out = qlora_train.normalize(row)
    calls = out["messages"][0]["tool_calls"]
    assert calls[0]["function"]["arguments"] == {"file_path": "/x"}
    assert calls[1]["function"]["arguments"] == {"raw": "not json"}


def test_shadow_eval_execute_runs_commands():
    rcs = shadow_eval.run_commands(["true", "false"], execute=True)
    assert rcs == [0, 1]
    assert shadow_eval.run_commands(["false"], execute=False) == []


def test_promote_candidate_proposal_leaves_config_untouched(tmp_path):
    # Spec: promotion emits a proposed diff for human review; in-place edit
    # only via --apply (autonomy earned, umbrella principle 1).
    cfg = tmp_path / "harness.toml"
    original = (
        '[[backends]]\nname = "cand"\nbase_url = "http://cand/v1"\n'
        'model = "c"\nroles = ["candidate"]\n'
    )
    cfg.write_text(original)
    diff = promote_candidate.propose_config(cfg, "cand", ["main", "subagent"])
    assert cfg.read_text() == original
    assert '+roles = ["main", "subagent"]' in diff
    proposed = cfg.with_suffix(".toml.proposed")
    assert 'roles = ["main", "subagent"]' in proposed.read_text()


def test_relax_scaffold_gate_and_config_edit(tmp_path):
    results = tmp_path / "results.jsonl"
    results.write_text("\n".join(json.dumps(r) for r in [
        {"model": "m", "invalid_calls": 0},
        {"model": "m", "invalid_calls": 0},
    ]))
    assert relax_scaffold.can_relax(results, "m", "invalid_calls", 0.0)
    cfg = tmp_path / "harness.toml"
    cfg.write_text(
        '[[backends]]\nname = "m"\nbase_url = "http://m/v1"\nmodel = "m"\nroles = ["main"]\n'
    )
    relax_scaffold.relax_config(cfg, "m", "guard_edit_without_read")
    assert 'relaxed = ["guard_edit_without_read"]' in cfg.read_text()
