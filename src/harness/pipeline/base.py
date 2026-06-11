from typing import Protocol, Sequence

from harness.config import Settings
from harness.ir import Conversation


class Stage(Protocol):
    def apply(self, conv: Conversation, settings: Settings) -> Conversation: ...


def run_pipeline(
    conv: Conversation, settings: Settings, stages: Sequence[Stage]
) -> Conversation:
    for stage in stages:
        conv = stage.apply(conv, settings)
    return conv
