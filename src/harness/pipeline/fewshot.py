"""Stage ⑤: append canonical tool-call examples to the system prompt.

Small models imitate concrete examples far more reliably than they follow
abstract rules; this is the cheapest single reliability win.
"""

from __future__ import annotations

from dataclasses import replace

from harness.config import Settings
from harness.ir import Conversation

HEADER = "## Tool call examples"

EXAMPLES = f"""\
{HEADER}

Example 1 — find then read:
  Task: "where is the retry limit configured?"
  Call Grep with {{"pattern": "retry_limit", "output_mode": "files_with_matches"}}
  Result: src/app/config.py
  Call Read with {{"file_path": "/repo/src/app/config.py"}}

Example 2 — edit after reading:
  Task: "raise the retry limit to 5"
  (file already read; it contains the line `RETRY_LIMIT = 3`)
  Call Edit with {{"file_path": "/repo/src/app/config.py", "old_string": "RETRY_LIMIT = 3", "new_string": "RETRY_LIMIT = 5"}}

Example 3 — verify:
  Call Bash with {{"command": "pytest tests/test_config.py -q", "description": "Run config tests"}}
  Result: 4 passed — task complete, stop and summarize.\
"""


class FewshotStage:
    def apply(self, conv: Conversation, settings: Settings) -> Conversation:
        if not settings.pipeline.fewshot or HEADER in conv.system:
            return conv
        return replace(conv, system=conv.system + "\n\n" + EXAMPLES)
