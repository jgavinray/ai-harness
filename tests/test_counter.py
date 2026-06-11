from harness.ir import Conversation, GenParams, TextPart, ToolDef, ToolResultPart, Turn
from harness.tokens.counter import HeuristicCounter, count_conversation


def test_heuristic_in_range():
    text = "hello world " * 100  # ~300 tokens for typical BPE
    n = HeuristicCounter().count_text(text)
    assert 180 <= n <= 420


def test_conversation_count_grows():
    counter = HeuristicCounter()
    base = Conversation(
        system="sys prompt",
        turns=(Turn("user", (TextPart("hi there"),)),),
        tools=(ToolDef("Read", "Reads a file", {"type": "object"}, {"type": "object"}),),
        params=GenParams(max_tokens=100),
    )
    bigger = Conversation(
        base.system,
        base.turns + (Turn("user", (ToolResultPart("t1", "x" * 4000),)),),
        base.tools,
        base.params,
    )
    a, b = count_conversation(base, counter), count_conversation(bigger, counter)
    assert 0 < a < b
    assert b - a > 900  # the 4000-char tool result dominates
