import httpx
import pytest

from harness.backends.base import BackendError
from harness.backends.pool import BackendPool, PooledBackend
from harness.config import BackendCfg, PoolBackendCfg, Settings
from harness.router import Router, session_key
from tests.fake_openai import FakeOpenAI, finish_chunk, text_chunk


def fleet_settings() -> Settings:
    s = Settings()
    s.backends = [
        PoolBackendCfg(name="big", kind="vllm", base_url="http://big/v1",
                       model="m35", profile="qwen", context_window=262144,
                       roles=["main"]),
        PoolBackendCfg(name="mid", kind="vllm", base_url="http://mid/v1",
                       model="m27", profile="qwen", context_window=131072,
                       roles=["subagent", "fast"]),
        PoolBackendCfg(name="gem", kind="llamacpp", base_url="http://gem/v1",
                       model="g31", profile="gemma", context_window=131072,
                       roles=["subagent"]),
    ]
    return s


def make_pool(s: Settings) -> BackendPool:
    client = httpx.AsyncClient()  # never actually used in routing tests
    return BackendPool(s, client)


def test_pool_from_fleet_config():
    pool = make_pool(fleet_settings())
    assert {b.name for b in pool.backends} == {"big", "mid", "gem"}
    big = pool.get("big")
    assert big.profile.name == "qwen"
    assert big.backend.constrained is True
    gem = pool.get("gem")
    assert gem.profile.name == "gemma"


def test_pool_single_backend_backcompat():
    s = Settings()  # no [[backends]] -> falls back to [backend]+[profile]
    pool = make_pool(s)
    assert len(pool.backends) == 1
    b = pool.backends[0]
    assert b.roles == ["main", "subagent", "fast"]
    assert b.profile.name == s.profile.name


def test_router_tier_haiku_goes_fast():
    s = fleet_settings()
    router = Router(make_pool(s), s)
    picked = router.pick({"model": "claude-haiku-4-5", "system": "x",
                          "messages": [{"role": "user", "content": "title this"}]})
    assert picked.name == "mid"


def test_router_main_then_overflow():
    s = fleet_settings()
    router = Router(make_pool(s), s)
    body1 = {"model": "claude-sonnet-4-6", "system": "envA",
             "messages": [{"role": "user", "content": "task one"}]}
    body2 = {"model": "claude-sonnet-4-6", "system": "envB",
             "messages": [{"role": "user", "content": "task two"}]}
    first = router.pick(body1)
    assert first.name == "big"
    first.in_flight += 1
    # different session while main is busy -> overflow to subagent role
    second = router.pick(body2)
    assert second.name in ("mid", "gem")


def test_router_session_affinity_beats_load():
    s = fleet_settings()
    router = Router(make_pool(s), s)
    body = {"model": "claude-sonnet-4-6", "system": "envZ",
            "messages": [{"role": "user", "content": "long task"}]}
    first = router.pick(body)
    first.in_flight += 5  # heavily loaded
    again = router.pick(body)  # same session key
    assert again.name == first.name  # affinity wins over load


def test_router_skips_cooled_down():
    s = fleet_settings()
    pool = make_pool(s)
    router = Router(pool, s)
    pool.get("big").trip(cooldown_s=60)
    picked = router.pick({"model": "claude-sonnet-4-6", "system": "q",
                          "messages": [{"role": "user", "content": "t"}]})
    assert picked.name != "big"


def test_session_key_stable_across_turns():
    sys_prompt = "You are Claude Code\n# Environment\nWorking directory: /repo\n"
    one_turn = {"system": sys_prompt, "messages": [{"role": "user", "content": "fix it"}]}
    many_turns = {"system": sys_prompt, "messages": [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "now the other file"},
    ]}
    assert session_key(one_turn) == session_key(many_turns)
    other = {"system": sys_prompt, "messages": [{"role": "user", "content": "different task"}]}
    assert session_key(other) != session_key(one_turn)


async def test_pool_marks_down_on_error_and_recovers():
    s = Settings()
    fake = FakeOpenAI()
    fake.push([{"_status": 500}])
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=fake.app), base_url="http://fake")
    s.backend.base_url = "http://fake/v1"
    pool = BackendPool(s, client)
    b = pool.backends[0]
    with pytest.raises(BackendError):
        async for _ in pool.stream(b, {"model": "m", "messages": []}):
            pass
    assert b.is_down() is True
    b.cooldown_until = 0  # simulate cooldown expiry
    fake.scripts.clear()
    fake.push([text_chunk("hi"), finish_chunk()])
    chunks = [c async for c in pool.stream(b, {"model": "m", "messages": []})]
    assert chunks and b.is_down() is False
