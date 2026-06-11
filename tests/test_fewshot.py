from harness.config import Settings
from harness.ir import Conversation, GenParams
from harness.pipeline.fewshot import FewshotStage


def conv() -> Conversation:
    return Conversation("base system", (), (), GenParams(max_tokens=10))


def test_examples_appended():
    out = FewshotStage().apply(conv(), Settings())
    assert "## Tool call examples" in out.system
    for needle in ("file_path", "old_string", "new_string", "command"):
        assert needle in out.system


def test_idempotent():
    s = Settings()
    once = FewshotStage().apply(conv(), s)
    twice = FewshotStage().apply(once, s)
    assert twice.system.count("## Tool call examples") == 1


def test_toggle_off():
    s = Settings()
    s.pipeline.fewshot = False
    assert FewshotStage().apply(conv(), s).system == "base system"
