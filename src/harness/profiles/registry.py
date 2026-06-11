from harness.profiles.base import Profile


class QwenProfile(Profile):
    name = "qwen"


class DeepseekR1Profile(Profile):
    name = "deepseek_r1"
    reasoning_tags = ("<think>", "</think>")


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
