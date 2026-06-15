from harness.config import Settings
from harness.guards import BAD_DEV_PR_PREFIX, GOOD_DEV_PR_PREFIX
from harness.ir import Conversation, GenParams, TextPart, ToolCallPart, ToolResultPart, Turn
from harness.pipeline.path_canon import PathCanonStage


def test_path_canon_rewrites_system_turns_results_and_tool_args():
    conv = Conversation(
        f"Working directory: {BAD_DEV_PR_PREFIX}",
        (
            Turn("user", (TextPart(f"Review {BAD_DEV_PR_PREFIX}/plan.md"),)),
            Turn("assistant", (ToolCallPart("r1", "Read", {"file_path": f"{BAD_DEV_PR_PREFIX}/src/main.c"}),)),
            Turn("user", (ToolResultPart("r1", f"No such file: {BAD_DEV_PR_PREFIX}/src/main.c", True),)),
        ),
        (),
        GenParams(max_tokens=512),
    )
    metrics: dict = {}
    out = PathCanonStage().apply(conv, Settings(), metrics)
    rendered = repr(out)
    assert BAD_DEV_PR_PREFIX not in rendered
    assert GOOD_DEV_PR_PREFIX in out.system
    assert out.turns[0].parts[0].text == f"Review {GOOD_DEV_PR_PREFIX}/plan.md"
    assert out.turns[1].parts[0].arguments["file_path"] == f"{GOOD_DEV_PR_PREFIX}/src/main.c"
    assert f"No such file: {GOOD_DEV_PR_PREFIX}/src/main.c" in out.turns[2].parts[0].content
    assert metrics["path_canonicalized"] is True


def test_path_canon_returns_identity_when_unchanged():
    conv = Conversation(
        "system",
        (Turn("user", (TextPart("hello"),)),),
        (),
        GenParams(max_tokens=512),
    )
    metrics: dict = {}
    out = PathCanonStage().apply(conv, Settings(), metrics)
    assert out is conv
    assert metrics["path_canonicalized"] is False
