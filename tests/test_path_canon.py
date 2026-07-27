from harness.config import Settings
from harness.ir import (
    Conversation,
    GenParams,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    Turn,
)
from harness.pipeline.path_canon import PathCanonStage

BAD = "/work/old-root"
GOOD = "/work/new-root"


def _settings() -> Settings:
    s = Settings()
    s.pipeline.path_aliases = [[BAD, GOOD]]
    return s


def test_path_canon_rewrites_system_turns_results_and_tool_args():
    conv = Conversation(
        f"Working directory: {BAD}",
        (
            Turn("user", (TextPart(f"Review {BAD}/plan.md"),)),
            Turn("assistant", (ToolCallPart("r1", "Read", {"file_path": f"{BAD}/src/main.c"}),)),
            Turn("user", (ToolResultPart("r1", f"No such file: {BAD}/src/main.c", True),)),
        ),
        (),
        GenParams(max_tokens=512),
    )
    metrics: dict = {}
    out = PathCanonStage().apply(conv, _settings(), metrics)
    rendered = repr(out)
    assert BAD not in rendered
    assert GOOD in out.system
    assert out.turns[0].parts[0].text == f"Review {GOOD}/plan.md"
    assert out.turns[1].parts[0].arguments["file_path"] == f"{GOOD}/src/main.c"
    assert f"No such file: {GOOD}/src/main.c" in out.turns[2].parts[0].content
    assert metrics["path_canonicalized"] is True


def test_path_canon_returns_identity_when_unchanged():
    conv = Conversation(
        "system",
        (Turn("user", (TextPart(f"hello from {BAD}/x"),)),),
        (),
        GenParams(max_tokens=512),
    )
    metrics: dict = {}
    out = PathCanonStage().apply(conv, Settings(), metrics)  # no aliases configured
    assert out is conv
    assert metrics["path_canonicalized"] is False
