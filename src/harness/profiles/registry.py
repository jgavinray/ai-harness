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


class DeepseekV4FlashProfile(Profile):
    # Verified 2026-07-29 against deepseek-v4-flash-dspark on vLLM 0.21.1rc1:
    # reasoning stays null on both trivial and step-by-step-forcing prompts
    # (chain-of-thought is written inline in `content`, not a separate
    # `reasoning`/`reasoning_content` delta field or <think>...</think> span),
    # and tool_calls follow the standard OpenAI id/type/function.arguments
    # shape. No overrides needed beyond the base Profile.
    name = "deepseek_v4_flash"


class DevstralProfile(Profile):
    name = "devstral"


class GemmaProfile(Profile):
    name = "gemma"
    supports_system_role = False


PROFILES: dict[str, Profile] = {
    p.name: p
    for p in (
        QwenProfile(),
        DeepseekR1Profile(),
        DeepseekV4FlashProfile(),
        DevstralProfile(),
        GemmaProfile(),
    )
}


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown profile {name!r}; available: {sorted(PROFILES)}") from None
