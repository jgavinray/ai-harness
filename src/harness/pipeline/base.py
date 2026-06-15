from inspect import signature
from typing import Protocol, Sequence

from harness.config import Settings
from harness.ir import Conversation


class Stage(Protocol):
    def apply(self, conv: Conversation, settings: Settings) -> Conversation: ...


def run_pipeline(
    conv: Conversation,
    settings: Settings,
    stages: Sequence[Stage],
    metrics: dict | None = None,
) -> Conversation:
    for stage in stages:
        if metrics is None:
            conv = stage.apply(conv, settings)
            continue
        if len(signature(stage.apply).parameters) >= 3:
            conv = stage.apply(conv, settings, metrics)  # type: ignore[misc]
        else:
            conv = stage.apply(conv, settings)
    return conv
