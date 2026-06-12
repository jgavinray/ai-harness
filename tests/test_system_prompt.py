from harness.config import Settings
from harness.ir import Conversation, GenParams
from harness.pipeline.system_prompt import SystemPromptStage

CC_SYSTEM = (
    "You are Claude Code, Anthropic's official CLI for Claude.\n\n"
    "You are an interactive CLI tool that helps users with software engineering tasks.\n\n"
    "# Tone and style\n" + ("Be concise. " * 300) + "\n\n"
    "# Tool usage policy\n" + ("Prefer Grep over bash grep. " * 300) + "\n\n"
    "# Doing tasks\n" + ("Plan carefully. " * 300) + "\n\n"
    "# Environment\nWorking directory: /home/user/project\nPlatform: linux\n\n"
    "# claudeMd\nContents of /home/user/project/CLAUDE.md:\nAlways run make lint before committing.\n"
)


def conv_with(system: str) -> Conversation:
    return Conversation(system, (), (), GenParams(max_tokens=100))


def settings(mode: str) -> Settings:
    s = Settings()
    s.pipeline.system_prompt = mode
    return s


def test_replace_mode_shrinks_and_keeps_context():
    out = SystemPromptStage().apply(conv_with(CC_SYSTEM), settings("replace"))
    assert len(out.system) < 4000 < len(CC_SYSTEM)
    # contract rules present
    assert "old_string" in out.system
    # environment + CLAUDE.md sections preserved verbatim
    assert "Working directory: /home/user/project" in out.system
    assert "Always run make lint before committing." in out.system
    # boilerplate gone
    assert "Be concise. Be concise." not in out.system


def test_non_cc_prompt_not_replaced():
    out = SystemPromptStage().apply(conv_with("My custom agent prompt"), settings("replace"))
    assert "My custom agent prompt" in out.system
    assert "old_string" not in out.system  # replacement not injected


def test_passthrough_identity():
    out = SystemPromptStage().apply(conv_with(CC_SYSTEM), settings("passthrough"))
    assert out.system == CC_SYSTEM


def test_compress_squeezes_whitespace():
    messy = "line1\n\n\n\n\nline2   \n"
    out = SystemPromptStage().apply(conv_with(messy), settings("compress"))
    assert "\n\n\n" not in out.system
    assert "line1" in out.system and "line2" in out.system


def test_replace_mode_frames_kept_context_as_stale_background():
    system = CC_SYSTEM + (
        "\n# memory\nLast session: investigating weather damage in fight.c "
        "at /Users/old/dev-pr. Change committed.\n"
    )
    out = SystemPromptStage().apply(conv_with(system), settings("replace"))
    # framing block present, after the contract but before every kept section
    framing_at = out.system.find("may be stale")
    assert framing_at != -1
    assert "the only task" in out.system
    assert out.system.find("old_string") < framing_at
    assert framing_at < out.system.find("Working directory: /home/user/project")
    assert framing_at < out.system.find("Last session: investigating weather damage")


def test_replace_mode_no_framing_when_nothing_kept():
    bare = "You are Claude Code, Anthropic's official CLI for Claude.\n\n# Tone and style\nBe concise.\n"
    out = SystemPromptStage().apply(conv_with(bare), settings("replace"))
    assert "may be stale" not in out.system


def test_sdk_print_mode_prompt_also_replaced():
    sdk_system = (
        "You are a Claude agent, built on Anthropic's Claude Agent SDK.\n\n"
        "# Tone and style\n" + ("Be concise. " * 300) + "\n\n"
        "# Environment\nWorking directory: /repo\n"
    )
    out = SystemPromptStage().apply(conv_with(sdk_system), settings("replace"))
    assert "old_string" in out.system
    assert "Working directory: /repo" in out.system
    assert len(out.system) < len(sdk_system)
