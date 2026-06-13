import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import lora_train  # noqa: E402
import promote_candidate  # noqa: E402
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
