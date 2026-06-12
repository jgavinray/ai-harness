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
        async for _ in pool.stream(b, {"model": "m", "messages": [], "stream": True}):
            pass
    assert b.is_down() is True
    b.cooldown_until = 0  # simulate cooldown expiry
    fake.scripts.clear()
    fake.push([text_chunk("hi"), finish_chunk()])
    chunks = [c async for c in pool.stream(b, {"model": "m", "messages": [], "stream": True})]
    assert chunks and b.is_down() is False


def test_reconfigure_updates_roles_and_preserves_counters():
    s = fleet_settings()
    pool = make_pool(s)
    big = pool.get("big")
    big.requests, big.errors = 7, 1
    big.prompt_tokens, big.cached_tokens = 1000, 400
    big.ttft_ms = [100, 200]

    s2 = fleet_settings()
    s2.backends[0].roles = ["main", "subagent"]
    s2.backends[0].context_window = 65536
    summary = pool.reconfigure(s2)

    assert pool.get("big") is big  # same object: counters survive by construction
    assert big.roles == ["main", "subagent"]
    assert big.cfg.context_window == 65536
    assert (big.requests, big.errors) == (7, 1)
    assert (big.prompt_tokens, big.cached_tokens) == (1000, 400)
    assert big.ttft_ms == [100, 200]
    assert summary["updated"] == ["big", "mid", "gem"]
    assert summary["added"] == [] and summary["removed"] == []


def test_reconfigure_adds_and_removes_backends():
    pool = make_pool(fleet_settings())
    s2 = fleet_settings()
    s2.backends = [
        s2.backends[0],
        PoolBackendCfg(name="new", base_url="http://new/v1", model="n1", roles=["fast"]),
    ]
    summary = pool.reconfigure(s2)
    assert [b.name for b in pool.backends] == ["big", "new"]
    assert summary["added"] == ["new"]
    assert summary["removed"] == ["gem", "mid"]


def test_request_role_main_cli():
    from harness.router import request_role
    body = {"model": "claude-opus-4-8",
            "system": "You are Claude Code, Anthropic's official CLI for Claude."}
    assert request_role(body) == "main"


def test_request_role_subagent_sdk():
    from harness.router import request_role
    body = {"model": "claude-opus-4-8",
            "system": [{"type": "text", "text": "You are a Claude agent, built on Anthropic's Claude Agent SDK."}]}
    assert request_role(body) == "subagent"


def test_request_role_haiku_fast():
    from harness.router import request_role
    assert request_role({"model": "claude-haiku-4-5"}) == "fast"


def test_request_role_unknown_defaults_main():
    from harness.router import request_role
    assert request_role({"model": "claude-opus-4-8", "system": "custom"}) == "main"


def test_router_routes_subagent_fingerprint_to_subagent_backend():
    pool = make_pool(fleet_settings())
    router = Router(pool, fleet_settings())
    body = {"model": "claude-opus-4-8",
            "system": "You are a Claude agent, built on Anthropic's Claude Agent SDK.",
            "messages": [{"role": "user", "content": "explore"}]}
    chosen = router.pick(body)
    assert chosen.name in ("mid", "gem")  # the subagent-role backends


def test_session_key_ignores_billing_header_block():
    # Claude Code prepends a billing-header text block whose content varies
    # per request; it must not change the session key or affinity breaks
    # (and trace rows lose their session grouping).
    base = [{"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."}]
    b1 = {"system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.173; cch=aaaa;"}] + base,
          "messages": [{"role": "user", "content": "task"}]}
    b2 = {"system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.173; cch=bbbb;"}] + base,
          "messages": [{"role": "user", "content": "task"}]}
    assert session_key(b1) == session_key(b2)


def capped_settings() -> Settings:
    s = fleet_settings()
    s.backends[0].max_in_flight = 1  # "big" (main role)
    return s


def test_backend_at_capacity_skipped():
    pool = make_pool(capped_settings())
    router = Router(pool, capped_settings())
    pool.get("big").in_flight = 1  # saturated
    body = {"model": "claude-opus-4-8",
            "system": "You are Claude Code, Anthropic's official CLI for Claude.",
            "messages": [{"role": "user", "content": "t1"}]}
    chosen = router.pick(body)
    assert chosen.name != "big"


def test_affinity_broken_when_at_capacity():
    pool = make_pool(capped_settings())
    router = Router(pool, capped_settings())
    body = {"model": "claude-opus-4-8",
            "system": "You are Claude Code, Anthropic's official CLI for Claude.",
            "messages": [{"role": "user", "content": "t2"}]}
    first = router.pick(body)
    assert first.name == "big"          # affinity established
    pool.get("big").in_flight = 1       # now saturated
    assert router.pick(body).name != "big"


def test_all_at_capacity_falls_back_to_least_loaded():
    s = fleet_settings()
    for b in s.backends:
        b.max_in_flight = 1
    pool = make_pool(s)
    router = Router(pool, s)
    for b in pool.backends:
        b.in_flight = 1
    pool.get("gem").in_flight = 1
    pool.get("mid").in_flight = 3
    body = {"model": "claude-opus-4-8",
            "system": "You are Claude Code, Anthropic's official CLI for Claude.",
            "messages": [{"role": "user", "content": "t3"}]}
    # nothing available: degrade gracefully to least-loaded, never error
    assert router.pick(body).in_flight == 1


def test_main_overflow_prefers_non_fast_backend():
    # Overflow from main must prefer a dense subagent backend (gem) over a
    # fast-role box (mid) when both are idle; fast-role hardware is the
    # cheap tier and the last resort for main-quality work.
    s = fleet_settings()
    pool = make_pool(s)
    router = Router(pool, s)
    pool.get("big").in_flight = 1  # main busy -> overflow
    body = {"model": "claude-opus-4-8",
            "system": "You are Claude Code, Anthropic's official CLI for Claude.",
            "messages": [{"role": "user", "content": "overflow me"}]}
    assert router.pick(body).name == "gem"


def test_overflow_avoids_host_of_busy_backend():
    # Two backends can share one physical box (qwen27 + gemma31 on .196).
    # Overflowing onto a backend that shares a host with the saturated one
    # adds prefill load to the same hardware instead of adding capacity;
    # an idle backend on another host wins even if it is fast-tier.
    s = fleet_settings()
    s.backends[0].base_url = "http://shared:8001/v1"  # big (main)
    s.backends[2].base_url = "http://shared:8000/v1"  # gem (subagent)
    pool = make_pool(s)
    router = Router(pool, s)
    pool.get("big").in_flight = 1  # main busy -> overflow
    body = {"model": "claude-opus-4-8",
            "system": "You are Claude Code, Anthropic's official CLI for Claude.",
            "messages": [{"role": "user", "content": "overflow me"}]}
    assert router.pick(body).name == "mid"


def test_overflow_does_not_steal_affinity():
    # A small-context request may overflow while its warm backend is
    # saturated, but the overflow target must not capture the session's
    # affinity: the session has to return to its KV-warm backend as soon
    # as capacity frees, not stay pinned to wherever overflow dumped it.
    pool = make_pool(capped_settings())
    router = Router(pool, capped_settings())
    body = {"model": "claude-opus-4-8",
            "system": "You are Claude Code, Anthropic's official CLI for Claude.",
            "messages": [{"role": "user", "content": "small task"}]}
    assert router.pick(body).name == "big"   # affinity established
    pool.get("big").in_flight = 1            # saturated
    assert router.pick(body).name != "big"   # overflows elsewhere
    pool.get("big").in_flight = 0            # capacity frees
    assert router.pick(body).name == "big"   # returns to the warm backend


def test_affinity_sticky_for_large_context_despite_capacity():
    # Re-prefilling a large context cold (40-120s observed) is far worse
    # than briefly queuing on the warm backend; big sessions never bounce.
    pool = make_pool(capped_settings())
    router = Router(pool, capped_settings())
    big_history = [{"role": "user", "content": "task"},
                   {"role": "assistant", "content": "x" * 80000},
                   {"role": "user", "content": "continue"}]
    body = {"model": "claude-opus-4-8",
            "system": "You are Claude Code, Anthropic's official CLI for Claude.",
            "messages": big_history}
    assert router.pick(body).name == "big"   # affinity established
    pool.get("big").in_flight = 1            # saturated
    assert router.pick(body).name == "big"   # large context: stays anyway
