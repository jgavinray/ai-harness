"""Deterministic workflow guards over conversation history.

These guards are deliberately plain software instead of model calls. They inspect
the already-rendered conversation state and return a named nudge plus feedback
text when the next model action would violate a workflow invariant. The relay
owns retry mechanics and metrics; this module owns only the guard predicates and
the feedback wording.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from harness.config import Settings
from harness.ir import Conversation, ToolCall, ToolCallPart
from harness.planning import plan_status

EDIT_TOOLS = {"Edit", "MultiEdit"}
VERIFY_WORDS = (
    "pytest",
    "test",
    "check",
    "npm test",
    "cargo test",
    "go test",
    "lint",
    "compile",
)
DONE_WORDS = ("done", "fixed", "complete", "completed", "implemented", "finished")
VERIFY_STEP_WORDS = ("verify", "test", "check", "run")
BUILD_WORDS = (
    "make",
    "cmake",
    "ninja",
    "cargo build",
    "npm run build",
    "go build",
    "gcc",
    "clang",
    "g++",
    "rustc",
    "ld ",
)
INSPECT_EXECUTABLES = {"cat", "grep", "rg", "sed", "ls", "find", "pwd", "head", "tail", "wc", "nl"}
NO_PROGRESS_EXECUTABLES = {"echo", "printf", "true", "false", ":"}
DANGEROUS_PATTERNS = (
    r"\brm\s+-rf\b",
    r"\bsudo\b",
    r"\bmkfs\b",
    r"\bdd\b.*\bof=",
    r"\bchmod\s+-R\s+777\b",
    r":\(\)\s*\{",
)
BAD_DEV_PR_PREFIX = "/Users/jgavinray/dev-pr"
GOOD_DEV_PR_PREFIX = "/Users/jgavinray/dev/pr"


@dataclass(frozen=True)
class PreflightDecision:
    decision: str
    call: ToolCall
    reason: str | None = None
    feedback: str | None = None
    original_arguments: dict | None = None
    rewritten_arguments: dict | None = None
    bash_command_class: str | None = None


def guard_metrics(metrics: dict) -> dict:
    fires = metrics.setdefault("guard_fires", {})
    if not isinstance(fires, dict):
        fires = {}
        metrics["guard_fires"] = fires
    return fires


def increment_guard(metrics: dict, name: str) -> None:
    fires = guard_metrics(metrics)
    fires[name] = fires.get(name, 0) + 1


def _file_arg(call: ToolCall | ToolCallPart) -> str:
    value = call.arguments.get("file_path") or call.arguments.get("path") or ""
    return str(value)

def normalize_confused_paths(call: ToolCall) -> tuple[ToolCall, bool]:
    """Fix the observed dev-pr/dev/pr path confusion before tool execution."""
    changed = False
    args = dict(call.arguments)
    for key in ("file_path", "path", "command"):
        value = args.get(key)
        if isinstance(value, str) and BAD_DEV_PR_PREFIX in value:
            args[key] = value.replace(BAD_DEV_PR_PREFIX, GOOD_DEV_PR_PREFIX)
            changed = True
    if not changed:
        return call, False
    return ToolCall(call.id, call.name, args, call.raw_arguments), True


def _has_tool(conv: Conversation, name: str) -> bool:
    return any(t.name == name for t in (*conv.tools, *conv.all_tools))


def _path_parent_missing(path: str) -> bool:
    if not path or not Path(path).is_absolute():
        return False
    return not Path(path).parent.exists()


def _outside_allowed_roots(path: str, settings: Settings) -> bool:
    if not settings.pipeline.allowed_roots or not Path(path).is_absolute():
        return False
    try:
        resolved = Path(path).resolve(strict=False)
        roots = [Path(root).resolve(strict=False) for root in settings.pipeline.allowed_roots]
    except OSError:
        return False
    return not any(resolved == root or root in resolved.parents for root in roots)


def _grep_alternation_without_extended(command: str) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    if not tokens or Path(tokens[0]).name != "grep":
        return None
    if any(t in {"-E", "--extended-regexp", "-P", "-F"} or "E" in t[1:] for t in tokens[1:] if t.startswith("-")):
        return None
    if not any("|" in t and t != "|" for t in tokens[1:]):
        return None
    leading = command[: len(command) - len(command.lstrip())]
    stripped = command.lstrip()
    if stripped.startswith("grep "):
        return leading + "grep -E " + stripped[len("grep "):]
    return None


def _shell_redirect_path(command: str) -> str | None:
    match = re.search(r"(?:^|\s)(?:>|>>)\s*(['\"]?)([^'\"\s]+)\1", command)
    return match.group(2) if match else None


def classify_bash_command(command: str) -> str:
    lowered = command.lower().strip()
    if any(re.search(pattern, lowered) for pattern in DANGEROUS_PATTERNS):
        return "dangerous"
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = []
    executable = Path(tokens[0]).name if tokens else ""
    if executable in NO_PROGRESS_EXECUTABLES:
        return "no_progress"
    if executable == "mkdir":
        return "create_dir"
    if executable == "git" and len(tokens) > 2 and tokens[1] == "diff" and "--check" in tokens[2:]:
        return "verify"
    if any(word in lowered for word in BUILD_WORDS):
        return "build"
    if any(word in lowered for word in ("pytest", "npm test", "cargo test", "go test")):
        return "test"
    if any(word in lowered for word in VERIFY_WORDS):
        return "verify"
    if executable in INSPECT_EXECUTABLES:
        return "inspect"
    if executable == "git" and len(tokens) > 1 and tokens[1] in {
        "branch",
        "status",
        "rev-parse",
        "log",
        "show",
        "worktree",
    }:
        return "inspect"
    return "unknown"


def _latest_user_text(conv: Conversation) -> str:
    for turn in reversed(conv.turns):
        if turn.role != "user":
            continue
        texts = []
        for part in turn.parts:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                texts.append(text)
        if texts:
            return "\n".join(texts).lower()
    return ""


def _has_verify_intent(text: str) -> bool:
    return bool(
        re.search(
            r"\b(verify|check|build|compile|lint)\b|\brun(?:ning)?\s+(?:the\s+)?tests?\b",
            text,
        )
    )


def _verification_required(conv: Conversation, settings: Settings) -> bool:
    if has_unverified_edit(conv):
        return True
    plan = plan_status(conv.system)
    if settings.planning.enabled and plan is not None and _is_verify_step(plan[2]):
        return True
    latest = _latest_user_text(conv)
    return _has_verify_intent(latest)


def _first_path_token(tokens: list[str]) -> str | None:
    for token in tokens[1:]:
        if not token or token.startswith("-"):
            continue
        if token.isdigit():
            continue
        if token in {"|", "&&", "||", ";"}:
            continue
        if "/" in token or Path(token).suffix:
            return token
    return None


def _structured_tool_feedback(call: ToolCall, conv: Conversation) -> tuple[str, str] | None:
    if call.name != "Bash":
        return None
    command = str(call.arguments.get("command") or "").strip()
    if not command:
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    if not tokens:
        return None
    executable = Path(tokens[0]).name
    if executable in {"cat", "head", "tail", "sed", "wc", "nl"} and _has_tool(conv, "Read"):
        path = _first_path_token(tokens)
        if path is None and len(tokens) >= 2:
            path = tokens[-1]
        return (
            "use_read_tool",
            f"Use the Read tool for file inspection instead of Bash {executable}. "
            f"Call Read with file_path={path!r}.",
        )
    if executable in {"grep", "rg"} and _has_tool(conv, "Grep"):
        return (
            "use_grep_tool",
            "Use the Grep tool for source searches instead of Bash grep/rg. "
            "Call Grep with the pattern and path arguments.",
        )
    return None


def _repeated_failing_call(conv: Conversation, call: ToolCall) -> bool:
    if call.name == "Bash" and classify_bash_command(str(call.arguments.get("command") or "")) in {
        "build",
        "test",
        "verify",
    }:
        return False
    turns = list(conv.turns)
    last_match = -1
    for idx, turn in enumerate(turns):
        if turn.role != "assistant":
            continue
        for part in turn.parts:
            if (
                isinstance(part, ToolCallPart)
                and part.name == call.name
                and part.arguments == call.arguments
            ):
                last_match = idx
    if last_match < 0 or last_match + 1 >= len(turns):
        return False
    for turn in turns[last_match + 1 :]:
        for part in turn.parts:
            if isinstance(part, ToolCallPart) and part.name in EDIT_TOOLS | {"Write"}:
                return False
    return any(getattr(part, "is_error", False) for part in turns[last_match + 1].parts)


def preflight_tool_call(
    conv: Conversation, call: ToolCall, settings: Settings
) -> PreflightDecision:
    """Validate or correct a repaired tool call before it reaches the client."""
    original = dict(call.arguments)
    bash_class: str | None = None
    external_policy = settings.pipeline.policy_owner == "agentic_os"

    rewritten, path_rewritten = normalize_confused_paths(call)
    if path_rewritten:
        return PreflightDecision(
            "rewrite",
            rewritten,
            "path_alias",
            original_arguments=original,
            rewritten_arguments=dict(rewritten.arguments),
        )
    call = rewritten
    if not external_policy and _repeated_failing_call(conv, call):
        return PreflightDecision(
            "deny",
            call,
            "repeated_failing_call",
            "This exact tool call already failed and no relevant edit occurred afterward. "
            "Use the previous error, change the arguments, or take a different action.",
            original_arguments=original,
        )

    for key in ("file_path", "path"):
        value = call.arguments.get(key)
        if not isinstance(value, str):
            continue
        if _outside_allowed_roots(value, settings):
            return PreflightDecision(
                "deny",
                call,
                "outside_allowed_roots",
                f"{value!r} is outside the configured allowed roots. Choose a path under an allowed root.",
                original_arguments=original,
            )
        if call.name == "Write" and _path_parent_missing(value):
            return PreflightDecision(
                "deny",
                call,
                "missing_parent",
                f"The parent directory for {value!r} does not exist. Create the directory first, then retry Write.",
                original_arguments=original,
            )

    if call.name == "Bash":
        command = str(call.arguments.get("command") or "")
        bash_class = classify_bash_command(command)
        if bash_class == "dangerous":
            return PreflightDecision(
                "deny",
                call,
                "dangerous_command",
                "This Bash command is classified as dangerous and will not be run by the harness.",
                original_arguments=original,
                bash_command_class=bash_class,
            )
        fixed_grep = _grep_alternation_without_extended(command)
        if fixed_grep:
            args = dict(call.arguments)
            args["command"] = fixed_grep
            rewritten = ToolCall(call.id, call.name, args, call.raw_arguments)
            return PreflightDecision(
                "rewrite",
                rewritten,
                "grep_extended_regexp",
                original_arguments=original,
                rewritten_arguments=dict(rewritten.arguments),
                bash_command_class=bash_class,
            )
        redirect_path = _shell_redirect_path(command)
        if redirect_path and _path_parent_missing(redirect_path):
            return PreflightDecision(
                "deny",
                call,
                "missing_parent",
                f"The shell redirection target {redirect_path!r} has no existing parent directory. "
                "Create the parent directory first.",
                original_arguments=original,
                bash_command_class=bash_class,
            )
        if (
            not external_policy
            and _verification_required(conv, settings)
            and bash_class not in {"build", "test", "verify"}
        ):
            return PreflightDecision(
                "deny",
                call,
                "non_verification_command",
                "The current runtime state requires real verification after an edit or verify request. "
                "Run a project test, build, compile, lint, or check command. Do not use no-op or "
                "inspection commands such as echo, pwd, ls, head, wc, or git branch as verification.",
                original_arguments=original,
                bash_command_class=bash_class,
            )
        if not external_policy:
            structured = _structured_tool_feedback(call, conv)
            if structured is not None:
                reason, feedback = structured
                return PreflightDecision(
                    "deny",
                    call,
                    reason,
                    feedback,
                    original_arguments=original,
                    bash_command_class=bash_class,
                )

    return PreflightDecision("allow", call, original_arguments=original, bash_command_class=bash_class)

def _read_files(conv: Conversation) -> set[str]:
    out: set[str] = set()
    for turn in conv.turns:
        for part in turn.parts:
            if isinstance(part, ToolCallPart) and part.name == "Read":
                path = _file_arg(part)
                if path:
                    out.add(path)
    return out

def is_verification_command(command: str) -> bool:
    return classify_bash_command(command) in {"build", "test", "verify"}

def has_unverified_edit(conv: Conversation) -> bool:
    edited = False
    for turn in conv.turns:
        for part in turn.parts:
            if not isinstance(part, ToolCallPart):
                continue
            if part.name in EDIT_TOOLS or part.name == "Write":
                edited = True
            elif part.name == "Bash" and is_verification_command(
                str(part.arguments.get("command", ""))
            ):
                edited = False
    return edited

def _done_claim(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in DONE_WORDS)

def guard_tool_call(
    conv: Conversation, call: ToolCall, settings: Settings
) -> tuple[str, str] | None:
    if not settings.pipeline.workflow_guards:
        return None
    path = _file_arg(call)
    if (
        settings.pipeline.guard_edit_without_read
        and call.name in EDIT_TOOLS
        and path
        and path not in _read_files(conv)
    ):
        return (
            "edit_without_read",
            f"Read {path!r} before editing it, then retry the edit with the exact current text.",
        )
    plan = plan_status(conv.system)
    if (
        settings.planning.enabled
        and plan is not None
        and call.name in EDIT_TOOLS | {"Write"}
        and _is_verify_step(plan[2])
    ):
        return (
            "plan_drift",
            f"The current plan step is verification: {plan[2]!r}. "
            "Run the verification step before making more edits unless verification fails.",
        )
    return None

def guard_done_claim(
    conv: Conversation, text: str, settings: Settings
) -> tuple[str, str] | None:
    if not settings.pipeline.workflow_guards or not settings.pipeline.guard_verify_after_edit:
        return None
    plan = plan_status(conv.system)
    if settings.planning.enabled and plan is not None and _done_claim(text) and plan[0] < plan[1]:
        return (
            "plan_drift",
            f"The plan still has open steps: currently step {plan[0]}/{plan[1]} ({plan[2]}). "
            "Continue with the next planned action instead of claiming completion.",
        )
    if has_unverified_edit(conv) and _done_claim(text):
        return (
            "verify_after_edit",
            "You changed files but have not run a relevant test or check since the edit. "
            "Run a verification command now; only claim completion after it passes.",
        )
    return None

def _is_verify_step(step: str) -> bool:
    return _has_verify_intent(step.lower())
