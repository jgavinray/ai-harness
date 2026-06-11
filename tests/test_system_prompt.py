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
