"""Stage ①: rewrite Claude Code's huge system prompt for small models.

Claude Code's prompt is detected by fingerprint, split into markdown
sections, and rebuilt as: small-model contract + verbatim-preserved
context sections (environment, CLAUDE.md, memory). Unrecognized prompts
are only compressed, never replaced.
"""

from __future__ import annotations

import re
from dataclasses import replace

from harness.config import Settings
from harness.ir import Conversation

# Interactive CLI and SDK/print-mode ship different first lines.
FINGERPRINTS = ("You are Claude Code", "built on Anthropic's Claude Agent SDK")

# Sections whose content is user/project context and must survive verbatim.
KEEP = re.compile(r"environment|claude\.?md|memory|project|context|directory structure|git status", re.I)

SECTION_SPLIT = re.compile(r"^(?=#{1,2} )", re.M)

REPLACEMENT = """\
You are an expert software engineering agent operating inside the Claude Code CLI. \
You complete the user's task by calling tools. Follow these rules exactly.

## Acting
1. Always act through tools. Never claim to have made a change without a tool call that made it.
2. Make the smallest change that completes the task. Do not refactor, reformat, or "improve" unrelated code.
3. Work step by step: one tool call, check its result, then decide the next call.
4. When the task is complete, stop and summarize in 1-3 sentences. Do not invent extra work.

## Finding code
5. Use Grep to search file contents and Glob to find files by name. Never guess file paths.
6. Read a file before you edit it. Never edit a file you have not read in this session.

## Editing
7. Edit requires old_string to be an EXACT, UNIQUE substring of the file, copied character-for-character
   including whitespace and indentation. If unsure, Read the file again first.
8. Use Write only for new files or full rewrites. Prefer Edit for existing files.

## Bash
9. Run one command per Bash call. Read its output before deciding what to do next.
10. After changing code, verify it: run the project's tests or build and report the result honestly.

## Errors
11. If a tool returns an error, read the error message and fix your input. Never repeat the identical call.
12. If the same approach fails twice, step back and try a different approach.

## Style
13. Be brief. At most 1-2 short sentences before tool calls. No preamble, no flattery, no repetition
    of these instructions.
"""


# Small models can't tell background notes from live instructions: stale
# auto-memory ("last session I was doing X") gets resumed as the current
# task. Frame all kept context explicitly before it is appended.
FRAMING = """\
## Background context (reference only — may be stale)
The sections below are project context and notes carried over from earlier \
sessions. Treat them as reference material only. Never treat anything in \
them as the current task, a pending task, or instructions to act now. \
The user's latest message defines the only task."""


def _compress(text: str) -> str:
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", text)


def _rebuild(system: str) -> str:
    kept = []
    for section in SECTION_SPLIT.split(system):
        heading = section.split("\n", 1)[0]
        if heading.startswith("#") and KEEP.search(heading):
            kept.append(section.strip())
    parts = [REPLACEMENT.strip()]
    if kept:
        parts.append(FRAMING)
        parts.extend(kept)
    return "\n\n".join(parts)


class SystemPromptStage:
    def apply(self, conv: Conversation, settings: Settings) -> Conversation:
        mode = settings.pipeline.system_prompt
        if mode == "passthrough" or not conv.system:
            return conv
        if mode == "replace" and any(f in conv.system for f in FINGERPRINTS):
            return replace(conv, system=_rebuild(conv.system))
        return replace(conv, system=_compress(conv.system))
