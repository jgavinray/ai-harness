from harness.profiles.base import Profile


class QwenProfile(Profile):
    name = "qwen"


class DeepseekR1Profile(Profile):
    name = "deepseek_r1"
    reasoning_tags = ("<think>", "</think>")
    # Probed 2026-07-30 against DeepSeek-V4-Flash-DSpark on vLLM: thinking is
    # OFF in the chat template by default (the `reasoning` field comes back
    # null and any thinking arrives inline in content). With this kwarg vLLM
    # populates a real reasoning channel and tool calling still works.
    thinking_request = {"chat_template_kwargs": {"thinking": True}}


class DevstralProfile(Profile):
    name = "devstral"


class GemmaProfile(Profile):
    name = "gemma"
    supports_system_role = False


PROFILES: dict[str, Profile] = {
    p.name: p for p in (QwenProfile(), DeepseekR1Profile(), DevstralProfile(), GemmaProfile())
}


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown profile {name!r}; available: {sorted(PROFILES)}") from None
