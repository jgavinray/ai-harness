from harness.config import Settings, load_settings


def test_defaults():
    s = Settings()
    assert s.server.port == 8484
    assert s.pipeline.system_prompt == "replace"
    assert s.pipeline.max_tools == 8
    assert s.pipeline.repair_retries == 2
    assert s.pipeline.compact_at_ratio == 0.80
    assert s.pipeline.compact_target_ratio == 0.50
    assert s.pipeline.action_state_tools is True
    assert s.profile.context_window == 32768


def test_load_toml(tmp_path):
    p = tmp_path / "harness.toml"
    p.write_text('[backend]\nmodel = "qwen2.5-coder:32b"\n[profile]\nname = "qwen"\n')
    s = load_settings(p)
    assert s.backend.model == "qwen2.5-coder:32b"
    assert s.profile.name == "qwen"
    # untouched sections keep defaults
    assert s.server.port == 8484


def test_load_missing_path_gives_defaults(tmp_path):
    s = load_settings(tmp_path / "nope.toml")
    assert s.backend.kind == "openai"
