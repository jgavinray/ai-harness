import httpx

from harness.config import Settings
from harness.ir import (
    Conversation,
    Done,
    GenParams,
    TextDelta,
    TextPart,
    ToolCall,
    ToolDef,
    ToolCallPart,
    ToolResultPart,
    Turn,
)
from harness.profiles.registry import get_profile
from harness.relay import run
from tests.fake_openai import FakeOpenAI, finish_chunk, text_chunk, tool_chunk
from tests.test_backends import make

READ_SCHEMA = {
    "type": "object",
    "properties": {"file_path": {"type": "string"}},
    "required": ["file_path"],
}
EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string"},
        "old_string": {"type": "string"},
        "new_string": {"type": "string"},
    },
    "required": ["file_path", "old_string", "new_string"],
}
BASH_SCHEMA = {
    "type": "object",
    "properties": {"command": {"type": "string"}},
    "required": ["command"],
}
GREP_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string"},
        "path": {"type": "string"},
    },
    "required": ["pattern"],
}
SKILL_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
}


def conv() -> Conversation:
    return Conversation(
        "sys",
        (Turn("user", (TextPart("read x"),)),),
        (ToolDef("Read", "reads", READ_SCHEMA, READ_SCHEMA),),
        GenParams(max_tokens=512, stream=True),
    )


async def collect_events(fake: FakeOpenAI, kind: str = "openai", settings: Settings | None = None):
    settings = settings or Settings()
    backend = make(fake, kind)
    return [e async for e in run(conv(), get_profile("qwen"), backend, settings)]


async def test_happy_path():
    fake = FakeOpenAI()
    fake.push([
        text_chunk("ok"),
        tool_chunk("c1", "Read", '{"file_path": "/x"}'),
        finish_chunk("tool_calls"),
    ])
    evs = await collect_events(fake)
    assert TextDelta("ok") not in evs
    assert any(isinstance(e, ToolCall) and e.arguments == {"file_path": "/x"} for e in evs)
    assert evs[-1].stop_reason == "tool_use"
    assert len(fake.requests) == 1


async def test_dev_pr_path_confusion_is_rewritten_before_tool_call():
    fake = FakeOpenAI()
    fake.push([
        tool_chunk("c1", "Read", '{"file_path": "/Users/jgavinray/dev-pr/src/main.c"}'),
        finish_chunk("tool_calls"),
    ])
    metrics = {}
    backend = make(fake)
    evs = [e async for e in run(conv(), get_profile("qwen"), backend, Settings(), metrics=metrics)]
    call = next(e for e in evs if isinstance(e, ToolCall))
    assert call.arguments["file_path"] == "/Users/jgavinray/dev/pr/src/main.c"
    assert metrics["path_rewrites"] == 1
    assert metrics["path_rewrite_names"] == ["Read"]
    assert metrics["preflight_rewrites"] == 1
    assert metrics["preflight_reason"] == "path_alias"
    assert metrics["preflight_events"][0]["original_arguments"]["file_path"].startswith(
        "/Users/jgavinray/dev-pr"
    )


async def test_bad_then_good_retries_with_feedback():
    fake = FakeOpenAI()
    fake.push([
        text_chunk("trying"),
        tool_chunk("c1", "Read", '{"wrong_param": 1}'),
        finish_chunk("tool_calls"),
    ])
    fake.push([
        text_chunk("retry noise"),
        tool_chunk("c2", "Read", '{"file_path": "/x"}'),
        finish_chunk("tool_calls"),
    ])
    fake.push([finish_chunk("stop")])  # safety
    evs = await collect_events(fake)
    assert len(fake.requests) == 2
    # feedback message present in the second request
    second_msgs = fake.requests[1]["messages"]
    assert any("file_path" in str(m.get("content")) and m["role"] == "user" for m in second_msgs)
    # retry text suppressed, valid call emitted
    assert TextDelta("retry noise") not in evs
    assert any(isinstance(e, ToolCall) and e.arguments == {"file_path": "/x"} for e in evs)


async def test_invalid_tool_event_logs_full_raw_arguments():
    fake = FakeOpenAI()
    raw = '{"wrong_param": "' + ("x" * 700) + '"}'
    fake.push([
        tool_chunk("c1", "Read", raw),
        finish_chunk("tool_calls"),
    ])
    fake.push([tool_chunk("c2", "Read", '{"file_path": "/x"}'), finish_chunk("tool_calls")])
    backend = make(fake)
    metrics: dict = {}
    evs = [e async for e in run(conv(), get_profile("qwen"), backend, Settings(), metrics=metrics)]
    assert any(isinstance(e, ToolCall) and e.arguments == {"file_path": "/x"} for e in evs)
    assert metrics["invalid_tool_events"][0]["arguments"]["wrong_param"] == "x" * 700
    assert len(metrics["invalid_tool_events"][0]["raw_arguments"]) > 500


async def test_tool_required_state_blocks_free_text():
    fake = FakeOpenAI()
    fake.push([text_chunk("I'll inspect it first."), finish_chunk("stop")])
    fake.push([tool_chunk("c1", "Read", '{"file_path": "/x"}'), finish_chunk("tool_calls")])
    backend = make(fake)
    metrics: dict = {}
    evs = [e async for e in run(conv(), get_profile("qwen"), backend, Settings(), metrics=metrics)]
    assert len(fake.requests) == 2
    assert TextDelta("I'll inspect it first.") not in evs
    assert any(isinstance(e, ToolCall) and e.name == "Read" for e in evs)
    assert metrics["action_state_blocks"] == 1
    assert "requires tool" in str(fake.requests[1])


async def test_retries_exhausted_degrades_to_text():
    fake = FakeOpenAI()
    bad = [tool_chunk("c1", "Read", '{"nope": 1}'), finish_chunk("tool_calls")]
    fake.push(bad)
    fake.push(bad)
    fake.push(bad)  # repeats forever
    evs = await collect_events(fake)
    assert len(fake.requests) == 3  # initial + 2 retries
    assert not any(isinstance(e, ToolCall) for e in evs)
    assert any(isinstance(e, TextDelta) and "invalid tool call" in e.text for e in evs)
    assert evs[-1].stop_reason == "end_turn"


async def test_constrained_backend_gets_schema_on_retry():
    fake = FakeOpenAI()
    fake.push([tool_chunk("c1", "Read", '{"nope": 1}'), finish_chunk("tool_calls")])
    fake.push([tool_chunk("c2", "Read", '{"file_path": "/x"}'), finish_chunk("tool_calls")])
    fake.push([finish_chunk("stop")])
    await collect_events(fake, kind="vllm")
    assert fake.requests[0]["guided_json"] == READ_SCHEMA
    assert fake.requests[1]["guided_json"] == READ_SCHEMA


async def test_degenerate_stream_aborted():
    fake = FakeOpenAI()
    fake.push([text_chunk("loop loop loop ")] * 300 + [finish_chunk("stop")])
    evs = await collect_events(fake)
    assert isinstance(evs[-1], Done)
    assert evs[-1].stop_reason == "end_turn"
    streamed = "".join(e.text for e in evs if isinstance(e, TextDelta))
    assert len(streamed) < 4500  # aborted well before 300 chunks


def conv_with_repeats(n: int) -> Conversation:
    from harness.ir import ToolCallPart, ToolResultPart
    turns: list[Turn] = [Turn("user", (TextPart("find the config"),))]
    for i in range(n):
        turns.append(Turn("assistant", (ToolCallPart(f"t{i}", "Read", {"file_path": "/x"}),)))
        turns.append(Turn("user", (ToolResultPart(f"t{i}", "same content"),)))
    return Conversation(
        "sys", tuple(turns),
        (ToolDef("Read", "reads", READ_SCHEMA, READ_SCHEMA),),
        GenParams(max_tokens=512, stream=True),
    )


async def test_cross_turn_loop_broken_with_feedback():
    import json
    # history already holds the identical call 3x; the 4th must trigger
    # loop-break feedback instead of being yielded
    fake = FakeOpenAI()
    fake.push([tool_chunk("c9", "Read", '{"file_path": "/x"}'), finish_chunk("tool_calls")])
    fake.push([text_chunk("the config is in /x; done"), finish_chunk("stop")])
    backend = make(fake, "openai")
    evs = [e async for e in run(conv_with_repeats(3), get_profile("qwen"), backend, Settings())]
    assert not any(isinstance(e, ToolCall) for e in evs)
    assert len(fake.requests) == 2
    assert "identical" in json.dumps(fake.requests[1])
    assert evs[-1].stop_reason == "end_turn"


def conv_with_bash_description_repeats(n: int) -> Conversation:
    turns: list[Turn] = [Turn("user", (TextPart("inspect directory"),))]
    for i in range(n):
        turns.append(Turn("assistant", (ToolCallPart(
            f"b{i}",
            "Bash",
            {"command": "ls -la /tmp", "description": f"check {i}"},
        ),)))
        turns.append(Turn("user", (ToolResultPart(f"b{i}", "same listing"),)))
    return Conversation(
        "sys", tuple(turns),
        (ToolDef("Bash", "runs", BASH_SCHEMA, BASH_SCHEMA),),
        GenParams(max_tokens=512, stream=True),
    )


async def test_cross_turn_loop_ignores_bash_description_metadata():
    fake = FakeOpenAI()
    fake.push([
        tool_chunk("b9", "Bash", "{\"command\": \"ls -la /tmp\", \"description\": \"check again\"}"),
        finish_chunk("tool_calls"),
    ])
    fake.push([text_chunk("the listing is already available"), finish_chunk("stop")])
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [
        e async for e in run(
            conv_with_bash_description_repeats(3),
            get_profile("qwen"),
            backend,
            Settings(),
            metrics,
        )
    ]
    assert not any(isinstance(e, ToolCall) for e in evs)
    assert len(fake.requests) == 2
    assert metrics["loop_breaks"] == 1
    assert "identical" in str(fake.requests[1])


async def test_cross_turn_loop_records_guard_fire():
    fake = FakeOpenAI()
    fake.push([tool_chunk("c9", "Read", '{"file_path": "/x"}'), finish_chunk("tool_calls")])
    fake.push([text_chunk("the config is in /x; done"), finish_chunk("stop")])
    backend = make(fake, "openai")
    metrics: dict = {}
    [e async for e in run(conv_with_repeats(3), get_profile("qwen"), backend, Settings(), metrics)]
    assert metrics["guard_fires"]["same_approach"] == 1


async def test_two_prior_repeats_pass_through():
    # re-running a command a couple of times is legitimate (e.g. pytest
    # after a fix); only sustained repetition is broken
    fake = FakeOpenAI()
    fake.push([tool_chunk("c9", "Read", '{"file_path": "/x"}'), finish_chunk("tool_calls")])
    backend = make(fake, "openai")
    evs = [e async for e in run(conv_with_repeats(2), get_profile("qwen"), backend, Settings())]
    assert any(isinstance(e, ToolCall) for e in evs)
    assert len(fake.requests) == 1


WEB_SCHEMA = {
    "type": "object",
    "properties": {"url": {"type": "string"}},
    "required": ["url"],
}


def conv_with_hidden_tool() -> Conversation:
    read = ToolDef("Read", "reads", READ_SCHEMA, READ_SCHEMA)
    web = ToolDef("WebFetch", "fetches a url", WEB_SCHEMA, WEB_SCHEMA)
    return Conversation(
        "sys",
        (Turn("user", (TextPart("fetch x"),)),),
        (read,),                      # only Read is surfaced
        GenParams(max_tokens=512, stream=True),
        all_tools=(read, web),        # WebFetch is catalog-only
    )


async def test_hidden_tool_valid_call_passes_through():
    # Model called a catalogued-but-unsurfaced tool with valid args:
    # zero-cost path, no retry round-trip.
    fake = FakeOpenAI()
    fake.push([tool_chunk("c1", "WebFetch", '{"url": "https://x"}'),
               finish_chunk("tool_calls")])
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [e async for e in run(conv_with_hidden_tool(), get_profile("qwen"),
                                backend, Settings(), metrics=metrics)]
    assert any(isinstance(e, ToolCall) and e.name == "WebFetch" for e in evs)
    assert len(fake.requests) == 1
    assert metrics["tool_surfaced"] == 1
    assert metrics["tool_surfaced_names"] == ["WebFetch"]


async def test_hidden_tool_invalid_call_swaps_schema_and_retries():
    fake = FakeOpenAI()
    fake.push([tool_chunk("c1", "WebFetch", '{"address": "x"}'),   # wrong param
               finish_chunk("tool_calls")])
    fake.push([tool_chunk("c2", "WebFetch", '{"url": "https://x"}'),
               finish_chunk("tool_calls")])
    fake.push([finish_chunk("stop")])  # safety
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [e async for e in run(conv_with_hidden_tool(), get_profile("qwen"),
                                backend, Settings(), metrics=metrics)]
    assert len(fake.requests) == 2
    # the retry request must offer the WebFetch schema
    retry_tools = [t["function"]["name"] for t in fake.requests[1].get("tools", [])]
    assert "WebFetch" in retry_tools
    # and the valid second call is emitted
    assert any(isinstance(e, ToolCall) and e.arguments == {"url": "https://x"} for e in evs)
    assert metrics["tool_surfaced"] == 1
    assert metrics["tool_surfaced_names"] == ["WebFetch"]


async def test_truly_unknown_tool_still_fails_with_feedback():
    # A tool in neither the surfaced set nor the catalog keeps today's
    # behavior: feedback retry, then degrade to text.
    fake = FakeOpenAI()
    bad = [tool_chunk("c1", "Nonexistent", '{"a": 1}'), finish_chunk("tool_calls")]
    fake.push(bad)
    fake.push(bad)
    fake.push(bad)
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [e async for e in run(conv_with_hidden_tool(), get_profile("qwen"),
                                backend, Settings(), metrics=metrics)]
    assert not any(isinstance(e, ToolCall) for e in evs)
    assert metrics["tool_surfaced"] == 0


def conv_with_edit_tools(turns=()) -> Conversation:
    return Conversation(
        "sys",
        tuple(turns) or (Turn("user", (TextPart("fix /x"),)),),
        (
            ToolDef("Read", "reads", READ_SCHEMA, READ_SCHEMA),
            ToolDef("Edit", "edits", EDIT_SCHEMA, EDIT_SCHEMA),
            ToolDef("Bash", "runs", BASH_SCHEMA, BASH_SCHEMA),
        ),
        GenParams(max_tokens=512, stream=True),
    )


def conv_with_search_tools() -> Conversation:
    return Conversation(
        "sys",
        (Turn("user", (TextPart("search source"),)),),
        (
            ToolDef("Bash", "runs", BASH_SCHEMA, BASH_SCHEMA),
            ToolDef("Grep", "searches", GREP_SCHEMA, GREP_SCHEMA),
        ),
        GenParams(max_tokens=512, stream=True),
    )


async def test_preflight_rewrites_grep_alternation():
    fake = FakeOpenAI()
    fake.push([
        tool_chunk("b1", "Bash", '{"command": "grep -rn \\"foo|bar\\" src"}'),
        finish_chunk("tool_calls"),
    ])
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [e async for e in run(conv_with_search_tools(), get_profile("qwen"), backend, Settings(), metrics)]
    call = next(e for e in evs if isinstance(e, ToolCall))
    assert call.arguments["command"] == 'grep -E -rn "foo|bar" src'
    assert metrics["preflight_rewrites"] == 1
    assert metrics["preflight_reason"] == "grep_extended_regexp"
    assert metrics["preflight_events"][0]["bash_command_class"] == "inspect"
    assert metrics["emitted_tool_calls"][0]["arguments"]["command"] == 'grep -E -rn "foo|bar" src'


async def test_preflight_denies_bash_cat_when_read_exists():
    read = ToolDef("Read", "reads", READ_SCHEMA, READ_SCHEMA)
    bash = ToolDef("Bash", "runs", BASH_SCHEMA, BASH_SCHEMA)
    conv = Conversation(
        "sys",
        (Turn("user", (TextPart("inspect /x"),)),),
        (read, bash),
        GenParams(max_tokens=512, stream=True),
    )
    fake = FakeOpenAI()
    fake.push([tool_chunk("b1", "Bash", '{"command": "cat /x"}'), finish_chunk("tool_calls")])
    fake.push([tool_chunk("r1", "Read", '{"file_path": "/x"}'), finish_chunk("tool_calls")])
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [e async for e in run(conv, get_profile("qwen"), backend, Settings(), metrics)]
    assert len(fake.requests) == 2
    assert not any(isinstance(e, ToolCall) and e.name == "Bash" for e in evs)
    assert any(isinstance(e, ToolCall) and e.name == "Read" for e in evs)
    assert metrics["preflight_denies"] == 1
    assert metrics["preflight_reasons"]["use_read_tool"] == 1
    assert "Use the Read tool" in str(fake.requests[1])


async def test_agentic_os_mode_allows_bash_cat_when_orchestrator_kept_it():
    read = ToolDef("Read", "reads", READ_SCHEMA, READ_SCHEMA)
    bash = ToolDef("Bash", "runs", BASH_SCHEMA, BASH_SCHEMA)
    conv = Conversation(
        "sys",
        (Turn("user", (TextPart("inspect /x"),)),),
        (read, bash),
        GenParams(max_tokens=512, stream=True),
    )
    fake = FakeOpenAI()
    fake.push([tool_chunk("b1", "Bash", '{"command": "cat /x"}'), finish_chunk("tool_calls")])
    backend = make(fake, "openai")
    settings = Settings()
    settings.pipeline.policy_owner = "agentic_os"
    metrics: dict = {}
    evs = [e async for e in run(conv, get_profile("qwen"), backend, settings, metrics)]
    assert len(fake.requests) == 1
    assert any(isinstance(e, ToolCall) and e.name == "Bash" for e in evs)
    assert metrics["preflight_denies"] == 0


async def test_preflight_denies_write_missing_parent(tmp_path):
    write_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["file_path", "content"],
    }
    write = ToolDef("Write", "writes", write_schema, write_schema)
    conv = Conversation(
        "sys",
        (Turn("user", (TextPart("create file"),)),),
        (write,),
        GenParams(max_tokens=512, stream=True),
    )
    missing = tmp_path / "missing" / "x.txt"
    fake = FakeOpenAI()
    fake.push([
        tool_chunk("w1", "Write", f'{{"file_path": "{missing}", "content": "x"}}'),
        finish_chunk("tool_calls"),
    ])
    fake.push([text_chunk("I need to create the directory first."), finish_chunk("stop")])
    backend = make(fake, "openai")
    settings = Settings()
    settings.pipeline.client_colocated = True
    metrics: dict = {}
    evs = [e async for e in run(conv, get_profile("qwen"), backend, settings, metrics)]
    assert not any(isinstance(e, ToolCall) and e.name == "Write" for e in evs)
    assert metrics["preflight_denies"] == 1
    assert metrics["preflight_reasons"]["missing_parent"] == 1
    assert "parent directory" in str(fake.requests[1])


async def test_missing_parent_requires_exact_absolute_mkdir_next(tmp_path):
    write_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["file_path", "content"],
    }
    write = ToolDef("Write", "writes", write_schema, write_schema)
    bash = ToolDef("Bash", "runs", BASH_SCHEMA, BASH_SCHEMA)
    missing = tmp_path / "missing" / "x.txt"
    parent = missing.parent
    conv = Conversation(
        "sys",
        (Turn("user", (TextPart("create file"),)),),
        (write, bash),
        GenParams(max_tokens=512, stream=True),
    )
    fake = FakeOpenAI()
    fake.push([
        tool_chunk("w1", "Write", f'{{"file_path": "{missing}", "content": "x"}}'),
        finish_chunk("tool_calls"),
    ])
    fake.push([
        tool_chunk("b1", "Bash", '{"command": "ls -la .", "description": "inspect"}'),
        finish_chunk("tool_calls"),
    ])
    fake.push([
        tool_chunk("b2", "Bash", f'{{"command": "mkdir -p {parent}", "description": "create parent"}}'),
        finish_chunk("tool_calls"),
    ])
    backend = make(fake, "openai")
    settings = Settings()
    settings.pipeline.client_colocated = True
    metrics: dict = {}
    evs = [e async for e in run(conv, get_profile("qwen"), backend, settings, metrics)]
    assert len(fake.requests) == 3
    assert not any(isinstance(e, ToolCall) and e.name == "Write" for e in evs)
    assert any(
        isinstance(e, ToolCall)
        and e.name == "Bash"
        and e.arguments["command"] == f"mkdir -p {parent}"
        for e in evs
    )
    assert metrics["preflight_denies"] == 2
    assert metrics["preflight_reasons"]["missing_parent"] == 1
    assert metrics["preflight_reasons"]["missing_parent_next_action"] == 1
    assert f"mkdir -p {parent}" in str(fake.requests[1])
    assert f"mkdir -p {parent}" in str(fake.requests[2])


async def test_preflight_denies_bash_heredoc_file_creation_even_in_agentic_os(tmp_path):
    write_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["file_path", "content"],
    }
    write = ToolDef("Write", "writes", write_schema, write_schema)
    bash = ToolDef("Bash", "runs", BASH_SCHEMA, BASH_SCHEMA)
    target = tmp_path / "x.sh"
    conv = Conversation(
        "sys",
        (Turn("user", (TextPart("create file"),)),),
        (write, bash),
        GenParams(max_tokens=512, stream=True),
    )
    fake = FakeOpenAI()
    fake.push([
        tool_chunk(
            "b1",
            "Bash",
            '{"command": "cat > /tmp/create_files.sh << \\"EOF\\"\\necho hi\\nEOF\\nbash /tmp/create_files.sh"}',
        ),
        finish_chunk("tool_calls"),
    ])
    fake.push([
        tool_chunk("w1", "Write", f'{{"file_path": "{target}", "content": "echo hi"}}'),
        finish_chunk("tool_calls"),
    ])
    settings = Settings()
    settings.pipeline.policy_owner = "agentic_os"
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [e async for e in run(conv, get_profile("qwen"), backend, settings, metrics)]
    assert len(fake.requests) == 2
    assert not any(isinstance(e, ToolCall) and e.name == "Bash" for e in evs)
    assert any(isinstance(e, ToolCall) and e.name == "Write" for e in evs)
    assert metrics["preflight_denies"] == 1
    assert metrics["preflight_reasons"]["use_write_tool"] == 1
    assert "call Write" in str(fake.requests[1])


async def test_remote_proxy_allows_write_missing_parent(tmp_path):
    """A non-colocated harness must not block Write on its own missing dirs.

    The client runs on a different host, so the parent existing (or not) on the
    harness filesystem is meaningless. With client_colocated False (default) the
    Write must pass straight through instead of deadlocking on missing_parent.
    """
    write_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["file_path", "content"],
    }
    write = ToolDef("Write", "writes", write_schema, write_schema)
    conv = Conversation(
        "sys",
        (Turn("user", (TextPart("create file"),)),),
        (write,),
        GenParams(max_tokens=512, stream=True),
    )
    missing = tmp_path / "missing" / "x.txt"
    fake = FakeOpenAI()
    fake.push([
        tool_chunk("w1", "Write", f'{{"file_path": "{missing}", "content": "x"}}'),
        finish_chunk("tool_calls"),
    ])
    settings = Settings()  # client_colocated defaults to False
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [e async for e in run(conv, get_profile("qwen"), backend, settings, metrics)]
    assert len(fake.requests) == 1
    assert any(isinstance(e, ToolCall) and e.name == "Write" for e in evs)
    assert metrics["preflight_denies"] == 0


async def test_remote_proxy_skips_missing_parent_next_action(tmp_path):
    """The forced-mkdir state machine must not engage on a non-colocated harness.

    Even if a prior turn carries a <missing_parent> marker, a non-colocated
    harness has no authority over the client filesystem, so it must not lock the
    model into an mkdir-only surface.
    """
    write_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["file_path", "content"],
    }
    write = ToolDef("Write", "writes", write_schema, write_schema)
    bash = ToolDef("Bash", "runs", BASH_SCHEMA, BASH_SCHEMA)
    missing = tmp_path / "missing" / "x.txt"
    conv = Conversation(
        "sys",
        (
            Turn("user", (TextPart("create file"),)),
            Turn(
                "user",
                (TextPart(f"<missing_parent>{missing.parent}</missing_parent>"),),
            ),
        ),
        (write, bash),
        GenParams(max_tokens=512, stream=True),
    )
    fake = FakeOpenAI()
    fake.push([
        tool_chunk("b1", "Bash", '{"command": "ls -la .", "description": "inspect"}'),
        finish_chunk("tool_calls"),
    ])
    settings = Settings()  # client_colocated defaults to False
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [e async for e in run(conv, get_profile("qwen"), backend, settings, metrics)]
    assert len(fake.requests) == 1
    assert any(isinstance(e, ToolCall) and e.name == "Bash" for e in evs)
    assert metrics["preflight_denies"] == 0


async def test_preflight_denial_uses_reviewer_feedback(tmp_path):
    write_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["file_path", "content"],
    }
    write = ToolDef("Write", "writes", write_schema, write_schema)
    bash = ToolDef("Bash", "runs", BASH_SCHEMA, BASH_SCHEMA)
    target = tmp_path / "x.sh"
    conv = Conversation(
        "sys",
        (Turn("user", (TextPart("create file"),)),),
        (write, bash),
        GenParams(max_tokens=512, stream=True),
    )
    fake = FakeOpenAI()
    fake.push([
        tool_chunk(
            "b1",
            "Bash",
            '{"command": "cat > /tmp/create_files.sh << \\"EOF\\"\\necho hi\\nEOF"}',
        ),
        finish_chunk("tool_calls"),
    ])
    fake.push([
        tool_chunk("w1", "Write", f'{{"file_path": "{target}", "content": "echo hi"}}'),
        finish_chunk("tool_calls"),
    ])
    seen: list[tuple[str, str]] = []

    async def reviewer(trigger, review_conv, message, review_metrics):
        seen.append((trigger, message))
        review_metrics["reviewer_called_by_test"] = True
        return "Use Write directly for file creation."

    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [
        e async for e in run(
            conv,
            get_profile("qwen"),
            backend,
            Settings(),
            metrics,
            reviewer=reviewer,
        )
    ]
    assert seen
    assert seen[0][0] == "use_write_tool"
    assert any(isinstance(e, ToolCall) and e.name == "Write" for e in evs)
    assert metrics["reviewer_called_by_test"] is True
    assert "Reviewer feedback" in str(fake.requests[1])
    assert "Use Write directly" in str(fake.requests[1])


async def test_preflight_denies_dangerous_bash():
    bash = ToolDef("Bash", "runs", BASH_SCHEMA, BASH_SCHEMA)
    conv = Conversation(
        "sys",
        (Turn("user", (TextPart("clean temp"),)),),
        (bash,),
        GenParams(max_tokens=512, stream=True),
    )
    fake = FakeOpenAI()
    fake.push([tool_chunk("b1", "Bash", '{"command": "rm -rf /tmp/something"}'), finish_chunk("tool_calls")])
    fake.push([text_chunk("I will use a safer command."), finish_chunk("stop")])
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [e async for e in run(conv, get_profile("qwen"), backend, Settings(), metrics)]
    assert not any(isinstance(e, ToolCall) for e in evs)
    assert metrics["preflight_denies"] == 1
    assert metrics["preflight_reasons"]["dangerous_command"] == 1
    assert metrics["preflight_events"][0]["bash_command_class"] == "dangerous"


async def test_preflight_denies_repeated_failing_call():
    turns = (
        Turn("user", (TextPart("read missing"),)),
        Turn("assistant", (ToolCallPart("r1", "Read", {"file_path": "/missing"}),)),
        Turn("user", (ToolResultPart("r1", "No such file", is_error=True),)),
    )
    read = ToolDef("Read", "reads", READ_SCHEMA, READ_SCHEMA)
    conv = Conversation("sys", turns, (read,), GenParams(max_tokens=512, stream=True))
    fake = FakeOpenAI()
    fake.push([tool_chunk("r2", "Read", '{"file_path": "/missing"}'), finish_chunk("tool_calls")])
    fake.push([text_chunk("I'll use the existing error."), finish_chunk("stop")])
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [e async for e in run(conv, get_profile("qwen"), backend, Settings(), metrics)]
    assert not any(isinstance(e, ToolCall) for e in evs)
    assert metrics["preflight_denies"] == 1
    assert metrics["preflight_reasons"]["repeated_failing_call"] == 1


async def test_edit_without_read_guard_retries_with_feedback():
    fake = FakeOpenAI()
    fake.push([
        tool_chunk("e1", "Edit", '{"file_path": "/x", "old_string": "a", "new_string": "b"}'),
        finish_chunk("tool_calls"),
    ])
    fake.push([tool_chunk("r1", "Read", '{"file_path": "/x"}'), finish_chunk("tool_calls")])
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [e async for e in run(conv_with_edit_tools(), get_profile("qwen"), backend, Settings(), metrics)]
    assert len(fake.requests) == 2
    assert not any(isinstance(e, ToolCall) and e.name == "Edit" for e in evs)
    assert any(isinstance(e, ToolCall) and e.name == "Read" for e in evs)
    assert "Read '/x' before editing" in str(fake.requests[1])
    assert metrics["guard_fires"]["edit_without_read"] == 1


async def test_edit_without_read_guard_can_be_disabled():
    fake = FakeOpenAI()
    fake.push([
        tool_chunk("e1", "Edit", '{"file_path": "/x", "old_string": "a", "new_string": "b"}'),
        finish_chunk("tool_calls"),
    ])
    s = Settings()
    s.pipeline.guard_edit_without_read = False
    backend = make(fake, "openai")
    evs = [e async for e in run(conv_with_edit_tools(), get_profile("qwen"), backend, s)]
    assert any(isinstance(e, ToolCall) and e.name == "Edit" for e in evs)
    assert len(fake.requests) == 1


def conv_after_unverified_edit() -> Conversation:
    turns = (
        Turn("user", (TextPart("fix /x"),)),
        Turn("assistant", (ToolCallPart("e1", "Edit", {
            "file_path": "/x", "old_string": "a", "new_string": "b",
        }),)),
        Turn("user", (ToolResultPart("e1", "edited"),)),
    )
    return conv_with_edit_tools(turns)


async def test_done_claim_after_edit_requires_verification():
    fake = FakeOpenAI()
    fake.push([text_chunk("done"), finish_chunk("stop")])
    fake.push([tool_chunk("b1", "Bash", '{"command": "python3 test_x.py"}'), finish_chunk("tool_calls")])
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [
        e async for e in run(
            conv_after_unverified_edit(), get_profile("qwen"), backend, Settings(), metrics
        )
    ]
    assert len(fake.requests) == 2
    assert TextDelta("done") not in evs
    assert any(isinstance(e, ToolCall) and e.name == "Bash" for e in evs)
    assert "have not run a relevant test" in str(fake.requests[1])
    assert metrics["guard_fires"]["verify_after_edit"] == 1


async def test_verify_state_denies_non_verification_bash():
    fake = FakeOpenAI()
    fake.push([
        tool_chunk("b1", "Bash", '{"command": "git branch --show-current"}'),
        finish_chunk("tool_calls"),
    ])
    fake.push([tool_chunk("b2", "Bash", '{"command": "make test"}'), finish_chunk("tool_calls")])
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [
        e async for e in run(
            conv_after_unverified_edit(), get_profile("qwen"), backend, Settings(), metrics
        )
    ]
    assert len(fake.requests) == 2
    assert not any(
        isinstance(e, ToolCall) and e.arguments.get("command") == "git branch --show-current"
        for e in evs
    )
    assert any(
        isinstance(e, ToolCall) and e.arguments.get("command") == "make test"
        for e in evs
    )
    assert metrics["preflight_denies"] == 1
    assert metrics["preflight_reasons"]["non_verification_command"] == 1
    assert metrics["preflight_events"][0]["bash_command_class"] == "inspect"
    assert "requires real verification" in str(fake.requests[1])


async def test_effort_testing_text_does_not_make_bash_verification_required():
    bash = ToolDef("Bash", "runs", BASH_SCHEMA, BASH_SCHEMA)
    conv = Conversation(
        "sys",
        (Turn("user", (TextPart("Set effort level to high: Comprehensive implementation with extensive testing and documentation"),)),),
        (bash,),
        GenParams(max_tokens=512, stream=True),
    )
    fake = FakeOpenAI()
    fake.push([tool_chunk("b1", "Bash", '{"command": "ls /tmp"}'), finish_chunk("tool_calls")])
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [e async for e in run(conv, get_profile("qwen"), backend, Settings(), metrics)]
    assert any(isinstance(e, ToolCall) and e.name == "Bash" for e in evs)
    assert metrics["preflight_denies"] == 0
    assert metrics["action_state"] == "inspect"


async def test_verify_state_accepts_compiler_build_command():
    fake = FakeOpenAI()
    fake.push([tool_chunk("b1", "Bash", '{"command": "gcc -o app main.o"}'), finish_chunk("tool_calls")])
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [
        e async for e in run(
            conv_after_unverified_edit(), get_profile("qwen"), backend, Settings(), metrics
        )
    ]
    assert any(isinstance(e, ToolCall) and e.arguments.get("command") == "gcc -o app main.o" for e in evs)
    assert metrics["preflight_denies"] == 0
    assert metrics["preflight_events"][0]["bash_command_class"] == "build"


async def test_verify_state_allows_read_only_inspection():
    fake = FakeOpenAI()
    fake.push([tool_chunk("r1", "Read", '{"file_path": "/x"}'), finish_chunk("tool_calls")])
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [
        e async for e in run(
            conv_after_unverified_edit(), get_profile("qwen"), backend, Settings(), metrics
        )
    ]
    assert len(fake.requests) == 1
    assert any(isinstance(e, ToolCall) and e.name == "Read" for e in evs)
    assert metrics["action_state_blocks"] == 0


async def test_verify_state_rejects_edit_even_if_model_emits_it():
    turns = (
        Turn("user", (TextPart("fix /x"),)),
        Turn("assistant", (ToolCallPart("r1", "Read", {"file_path": "/x"}),)),
        Turn("user", (ToolResultPart("r1", "contents"),)),
        Turn("assistant", (ToolCallPart("e0", "Edit", {
            "file_path": "/x", "old_string": "a", "new_string": "b",
        }),)),
        Turn("user", (ToolResultPart("e0", "edited"),)),
    )
    fake = FakeOpenAI()
    fake.push([
        tool_chunk("e1", "Edit", '{"file_path": "/x", "old_string": "a", "new_string": "b"}'),
        finish_chunk("tool_calls"),
    ])
    fake.push([tool_chunk("b1", "Bash", '{"command": "pytest -q"}'), finish_chunk("tool_calls")])
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [
        e async for e in run(
            conv_with_edit_tools(turns), get_profile("qwen"), backend, Settings(), metrics
        )
    ]
    assert len(fake.requests) == 2
    assert not any(isinstance(e, ToolCall) and e.name == "Edit" for e in evs)
    assert any(isinstance(e, ToolCall) and e.name == "Bash" for e in evs)
    assert metrics["action_state_blocks"] == 1
    assert "current runtime action state is 'verify'" in str(fake.requests[1])


async def test_edit_state_rejects_hidden_bash_even_if_pipeline_has_it():
    turns = (
        Turn("user", (TextPart("fix /x"),)),
        Turn("assistant", (ToolCallPart("r0", "Read", {"file_path": "/x"}),)),
        Turn("user", (ToolResultPart("r0", "old"),)),
    )
    fake = FakeOpenAI()
    fake.push([tool_chunk("b1", "Bash", '{"command": "grep -rn foo /x"}'), finish_chunk("tool_calls")])
    fake.push([
        tool_chunk("e1", "Edit", '{"file_path": "/x", "old_string": "old", "new_string": "new"}'),
        finish_chunk("tool_calls"),
    ])
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [
        e async for e in run(
            conv_with_edit_tools(turns), get_profile("qwen"), backend, Settings(), metrics
        )
    ]
    assert len(fake.requests) == 2
    assert not any(isinstance(e, ToolCall) and e.name == "Bash" for e in evs)
    assert any(isinstance(e, ToolCall) and e.name == "Edit" for e in evs)
    assert metrics["allowed_tools"] == ["Read", "Edit"]
    assert metrics["action_state_blocks"] == 1
    assert "Call one of these tools now: Read, Edit" in str(fake.requests[1])


async def test_preflight_denies_bash_head_when_read_exists():
    read = ToolDef("Read", "reads", READ_SCHEMA, READ_SCHEMA)
    bash = ToolDef("Bash", "runs", BASH_SCHEMA, BASH_SCHEMA)
    conv = Conversation(
        "sys",
        (Turn("user", (TextPart("inspect /plan.md"),)),),
        (read, bash),
        GenParams(max_tokens=512, stream=True),
    )
    fake = FakeOpenAI()
    fake.push([tool_chunk("b1", "Bash", '{"command": "head -121 /plan.md"}'), finish_chunk("tool_calls")])
    fake.push([tool_chunk("r1", "Read", '{"file_path": "/plan.md"}'), finish_chunk("tool_calls")])
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [e async for e in run(conv, get_profile("qwen"), backend, Settings(), metrics)]
    assert len(fake.requests) == 2
    assert not any(isinstance(e, ToolCall) and e.name == "Bash" for e in evs)
    assert any(isinstance(e, ToolCall) and e.name == "Read" for e in evs)
    assert metrics["preflight_denies"] == 1
    assert metrics["preflight_reasons"]["use_read_tool"] == 1
    assert "instead of Bash head" in str(fake.requests[1])


def conv_with_plan(system_status: str) -> Conversation:
    return Conversation(
        "sys\n\n## Execution plan\n1. Inspect\n2. Run tests\n3. Finish\n" + system_status,
        (
            Turn("user", (TextPart("continue"),)),
            Turn("assistant", (ToolCallPart("r1", "Read", {"file_path": "/x"}),)),
            Turn("user", (ToolResultPart("r1", "contents"),)),
        ),
        (
            ToolDef("Read", "reads", READ_SCHEMA, READ_SCHEMA),
            ToolDef("Edit", "edits", EDIT_SCHEMA, EDIT_SCHEMA),
            ToolDef("Bash", "runs", BASH_SCHEMA, BASH_SCHEMA),
        ),
        GenParams(max_tokens=512, stream=True),
    )


async def test_plan_done_claim_before_final_step_is_drift():
    fake = FakeOpenAI()
    fake.push([text_chunk("done"), finish_chunk("stop")])
    fake.push([tool_chunk("b1", "Bash", '{"command": "pytest -q"}'), finish_chunk("tool_calls")])
    s = Settings()
    s.planning.enabled = True
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [
        e async for e in run(
            conv_with_plan("Plan status: Step 2/3: Run tests; done: 1✓"),
            get_profile("qwen"), backend, s, metrics,
        )
    ]
    assert TextDelta("done") not in evs
    assert any(isinstance(e, ToolCall) and e.name == "Bash" for e in evs)
    assert metrics["guard_fires"]["plan_drift"] == 1
    assert metrics["plan_drift"] == 1


async def test_edit_during_verify_plan_step_is_drift():
    fake = FakeOpenAI()
    fake.push([
        tool_chunk("e1", "Edit", '{"file_path": "/x", "old_string": "a", "new_string": "b"}'),
        finish_chunk("tool_calls"),
    ])
    fake.push([tool_chunk("b1", "Bash", '{"command": "pytest -q"}'), finish_chunk("tool_calls")])
    s = Settings()
    s.planning.enabled = True
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [
        e async for e in run(
            conv_with_plan("Plan status: Step 2/3: Run tests; done: 1✓"),
            get_profile("qwen"), backend, s, metrics,
        )
    ]
    assert not any(isinstance(e, ToolCall) and e.name == "Edit" for e in evs)
    assert any(isinstance(e, ToolCall) and e.name == "Bash" for e in evs)
    assert metrics["guard_fires"]["plan_drift"] == 1
    assert metrics["plan_drift"] == 1


async def test_skill_call_injects_compiled_procedure(tmp_path):
    skills = tmp_path / "skills"
    cache = tmp_path / "cache"
    (skills / "review").mkdir(parents=True)
    (skills / "review" / "SKILL.md").write_text("- Inspect the diff\n- Report defects first\n")
    skill = ToolDef("Skill", "load a skill", SKILL_SCHEMA, SKILL_SCHEMA)
    read = ToolDef("Read", "reads", READ_SCHEMA, READ_SCHEMA)
    conv = Conversation(
        "sys",
        (Turn("user", (TextPart("use the review skill"),)),),
        (skill, read),
        GenParams(max_tokens=512, stream=True),
    )
    fake = FakeOpenAI()
    fake.push([tool_chunk("s1", "Skill", '{"name": "review"}'), finish_chunk("tool_calls")])
    fake.push([tool_chunk("r1", "Read", '{"file_path": "/x"}'), finish_chunk("tool_calls")])
    s = Settings()
    s.skills.enabled = True
    s.skills.dir = str(skills)
    s.skills.cache_dir = str(cache)
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [e async for e in run(conv, get_profile("qwen"), backend, s, metrics)]
    assert not any(isinstance(e, ToolCall) and e.name == "Skill" for e in evs)
    assert any(isinstance(e, ToolCall) and e.name == "Read" for e in evs)
    assert "Compiled skill procedure for review" in str(fake.requests[1])
    assert "Inspect the diff" in str(fake.requests[1])
    assert metrics["skill_compiled"] == 1


async def test_invalid_hidden_skill_forces_concrete_tool_retry(tmp_path):
    skills = tmp_path / "skills"
    skills.mkdir()
    read = ToolDef("Read", "reads", READ_SCHEMA, READ_SCHEMA)
    skill = ToolDef("Skill", "load a skill", SKILL_SCHEMA, SKILL_SCHEMA)
    conv = Conversation(
        "sys",
        (Turn("user", (TextPart("fix the failing test"),)),),
        (read,),
        GenParams(max_tokens=512, stream=True),
        all_tools=(read, skill),
    )
    fake = FakeOpenAI()
    fake.push([
        tool_chunk("s1", "Skill", '{"skill": "superpowers:systematic-debugging"}'),
        finish_chunk("tool_calls"),
    ])
    fake.push([text_chunk("I'll use that skill."), finish_chunk("stop")])
    fake.push([tool_chunk("r1", "Read", '{"file_path": "/x"}'), finish_chunk("tool_calls")])
    s = Settings()
    s.skills.enabled = True
    s.skills.dir = str(skills)
    backend = make(fake, "openai")
    metrics: dict = {}
    evs = [e async for e in run(conv, get_profile("qwen"), backend, s, metrics)]
    assert any(isinstance(e, ToolCall) and e.name == "Read" for e in evs)
    assert metrics["tool_surfaced"] == 1
    assert metrics["tool_surfaced_names"] == ["Skill"]
    assert "could not be validated by the harness" in str(fake.requests[1])
    assert "Your previous response still did not call a tool" in str(fake.requests[2])
