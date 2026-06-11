from harness.config import Settings
from harness.ir import Conversation, GenParams
from harness.memory import HEADER, MemoryManager, MemoryStage, project_key


def settings_with_memory(tmp_path, idle_s=0.0) -> Settings:
    s = Settings()
    s.memory.enabled = True
    s.memory.dir = str(tmp_path)
    s.memory.idle_s = idle_s
    return s


def test_project_key():
    assert project_key("# Environment\nWorking directory: /home/u/proj\n") == "home-u-proj"
    assert project_key("no env section here") == "default"


def test_merge_dedupes_and_caps(tmp_path):
    s = settings_with_memory(tmp_path)
    s.memory.max_chars = 200
    m = MemoryManager(s, None)
    m.merge("p", "- run make test\n- uses ruff\nnoise line ignored")
    m.merge("p", "- run make test\n- new fact")
    text = m.read("p")
    assert text.count("- run make test") == 1
    assert "- new fact" in text and "noise line" not in text
    m.merge("p", "\n".join(f"- filler fact number {i}" for i in range(20)))
    assert len(m.read("p")) <= 200


def test_stage_injects_memory(tmp_path):
    s = settings_with_memory(tmp_path)
    m = MemoryManager(s, None)
    m.merge("home-u-proj", "- always run make lint")
    conv = Conversation(
        "sys\nWorking directory: /home/u/proj", (), (), GenParams(max_tokens=10)
    )
    out = MemoryStage(m, s).apply(conv, s)
    assert HEADER in out.system and "- always run make lint" in out.system
    # idempotent
    again = MemoryStage(m, s).apply(out, s)
    assert again.system.count(HEADER) == 1


def test_stage_noop_when_disabled(tmp_path):
    s = settings_with_memory(tmp_path)
    s.memory.enabled = False
    m = MemoryManager(s, None)
    m.merge("default", "- fact")
    conv = Conversation("sys", (), (), GenParams(max_tokens=10))
    assert MemoryStage(m, s).apply(conv, s).system == "sys"


async def test_sweep_extracts_idle_sessions(tmp_path):
    s = settings_with_memory(tmp_path, idle_s=0.0)
    calls = []

    async def fake_completer(messages):
        calls.append(messages)
        return "- project uses pytest\n- build with make"

    m = MemoryManager(s, fake_completer)
    m.note("sess1", "Working directory: /repo", [
        {"role": "user", "content": "fix the test"},
        {"role": "assistant", "content": "done", "tool_calls": [
            {"function": {"name": "Bash", "arguments": '{"command": "pytest"}'}}]},
    ])
    await m.sweep()
    assert len(calls) == 1
    assert "fix the test" in calls[0][0]["content"]
    assert "- project uses pytest" in m.read("repo")
    assert m.sessions == {}  # consumed


async def test_sweep_survives_completer_failure(tmp_path):
    s = settings_with_memory(tmp_path, idle_s=0.0)

    async def boom(messages):
        raise RuntimeError("backend down")

    m = MemoryManager(s, boom)
    m.note("sess1", "Working directory: /repo", [{"role": "user", "content": "x"}])
    await m.sweep()  # must not raise
    assert m.read("repo") == ""
