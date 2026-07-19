"""Adversarial review debate: a hostile reviewer and the proposing side
argue over a candidate turn until consensus (spec
2026-07-19-adversarial-review-loop-design.md).

Invariants:
- debate text never enters the client-visible conversation; every debate
  prompt extends the executor's rendered conversation byte-for-byte so the
  backend's prefix cache absorbs the prefill;
- termination is consensus or a pathology valve (no-progress, client
  deadline) — never a round cap;
- any reviewer failure fails OPEN: the current candidate ships untouched;
- shadow callers ship the original events regardless of the outcome; only
  enforce callers ship ``DebateOutcome.events``;
- every round is logged to the reviews JSONL — the debate is also a sensor
  for mining future deterministic guard rules.

The reviewer and the counter-review both run on the ``review``-role backend
(same GPU, fresh logical context per round); regeneration goes through the
caller's ``regenerate`` callback, which re-runs the normal relay path.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, replace

from harness.config import Settings
from harness.ir import (
    Conversation,
    Done,
    IREvent,
    Ping,
    TextDelta,
    TextPart,
    ToolCall,
    ToolCallPart,
    Turn,
)
from harness.log import RequestLogger
from harness.review import _review_backend

REVIEWER_INSTRUCTION = (
    "[adversarial review] You are now a hostile reviewer of the assistant "
    "response directly above. Verify every claim and every tool-call "
    "argument strictly against evidence present earlier in this "
    "conversation and against the user's original request. You cannot read "
    "files: anything not evidenced in the conversation is unverified. Also "
    "reject responses that drift from what was actually asked.\n"
    "Reply with exactly one of:\n"
    "APPROVE — every claim is evidenced and in scope.\n"
    "FLAG: <one-line concern worth logging but not blocking>\n"
    "OBJECTION:\n<numbered, specific unsupported claims or scope drift, "
    "each citing the missing or contradicting evidence>\n"
    "Do not call tools. The first word of your reply must be APPROVE, FLAG "
    "or OBJECTION."
)

REVIEWER_REBUTTAL_INSTRUCTION = (
    "[adversarial review] The proposer rebutted your objection above. If "
    "the rebuttal cites convincing evidence from this conversation, reply "
    "APPROVE. Otherwise reply OBJECTION: followed by the objections that "
    "still stand. The first word of your reply must be APPROVE, FLAG or "
    "OBJECTION. Do not call tools."
)

COUNTER_INSTRUCTION = (
    "[adversarial review] A hostile reviewer rejected your response above:\n"
    "{objection}\n"
    "If the objection is wrong, reply REBUT: citing the exact evidence from "
    "this conversation that supports your response. If the objection is "
    "right, reply CONCEDE. The first word of your reply must be REBUT or "
    "CONCEDE. Do not call tools."
)

REGEN_FEEDBACK = (
    "[adversarial review] Your response above was rejected by a reviewer:\n"
    "{objection}\n"
    "Produce a corrected response now, using only evidence present in this "
    "conversation. Where evidence is missing, say what is missing instead "
    "of asserting."
)


@dataclass
class DebateOutcome:
    outcome: str  # consensus | deadlock | deadline | reviewer_error | skipped_no_backend
    rounds: int
    events: list[IREvent]  # the final candidate (enforce callers ship this)
    unresolved_objection: str | None = None


async def keepalive_iter(source, interval_s: float):
    """Yield the source's events, interleaving Ping whenever the source is
    silent for interval_s, so the client's SSE stream never idles."""
    it = source.__aiter__()
    while True:
        nxt = asyncio.ensure_future(it.__anext__())
        while not nxt.done():
            finished, _ = await asyncio.wait({nxt}, timeout=interval_s)
            if not finished:
                yield Ping()
        try:
            ev = nxt.result()
        except StopAsyncIteration:
            return
        yield ev


def _turn_from_events(events: list[IREvent]) -> Turn:
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    parts: list = []
    if text:
        parts.append(TextPart(text))
    for e in events:
        if isinstance(e, ToolCall):
            parts.append(ToolCallPart(e.id, e.name, e.arguments))
    if not parts:
        parts.append(TextPart(""))
    return Turn("assistant", tuple(parts))


def _candidate_hash(turn: Turn) -> str:
    blob = json.dumps(
        [
            (type(p).__name__, getattr(p, "text", None), getattr(p, "name", None),
             getattr(p, "arguments", None))
            for p in turn.parts
        ],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _objection_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def _done_of(events: list[IREvent]) -> Done | None:
    return next((e for e in reversed(events) if isinstance(e, Done)), None)


def _is_prose_only(events: list[IREvent]) -> bool:
    return not any(isinstance(e, ToolCall) for e in events)


_VERDICT_KINDS = {
    "APPROVE": "approve",
    "FLAG": "flag",
    "OBJECTION": "objection",
    "CONCEDE": "concede",
    "REBUT": "rebut",
}


def _parse_verdict(text: str) -> tuple[str, str]:
    """Returns (kind, body): approve | flag | objection | concede | rebut |
    malformed. Models preface and decorate their verdict (live shadow round
    2026-07-19), so the keyword is taken from the first line that LEADS with
    one — never from a mere mention inside prose — and body is everything
    after it."""
    for line in text.splitlines():
        candidate_line = line.strip().lstrip("*#->• ").rstrip("*")
        if not candidate_line:
            continue
        first = candidate_line.split(None, 1)[0]
        keyword = first.rstrip(":*").upper()
        if keyword in _VERDICT_KINDS:
            offset = text.index(line) + len(line)
            inline = candidate_line[len(first):].lstrip(" :*\n")
            rest = text[offset:].lstrip(" :\n")
            body = "\n".join(part for part in (inline, rest) if part)
            return (_VERDICT_KINDS[keyword], body)
    return ("malformed", text.strip())


class DebateManager:
    def __init__(self, settings: Settings, reviews_logger: RequestLogger | None = None) -> None:
        self.settings = settings
        self.cfg = settings.review
        self.reviews_logger = reviews_logger

    async def run(
        self,
        conv: Conversation,
        candidate_events: list[IREvent],
        pool,
        metrics: dict,
        *,
        backend=None,
        regenerate=None,
        session_key: str | None = None,
        parent_request_id: str | None = None,
        account_usage=None,
        original_shipped: bool = False,
    ) -> DebateOutcome:
        start = time.monotonic()
        reviewer_backend = _review_backend(pool)
        if reviewer_backend is None:
            return self._finish(
                DebateOutcome("skipped_no_backend", 0, candidate_events),
                metrics, start, session_key, parent_request_id,
            )

        original_events = candidate_events
        candidate_turn = _turn_from_events(candidate_events)
        exchange: tuple[str, str] | None = None  # (objection, rebuttal)
        seen_objections: set[str] = set()
        rounds = 0

        def _account_discarded(events: list[IREvent]) -> None:
            done = _done_of(events)
            if account_usage and backend is not None and done is not None:
                account_usage(backend, done, session_key)

        while True:
            if time.monotonic() - start >= self.cfg.client_deadline_s:
                unresolved = exchange[0] if exchange else None
                return self._finish(
                    DebateOutcome("deadline", rounds, candidate_events, unresolved),
                    metrics, start, session_key, parent_request_id,
                )
            rounds += 1
            round_record = {
                "kind": "debate_round",
                "round": rounds,
                "session_key": session_key,
                "parent_request_id": parent_request_id,
                "backend": reviewer_backend.name,
                "candidate_hash": _candidate_hash(candidate_turn),
                "counter": None,
                "objection": None,
                "rebuttal": None,
            }
            try:
                verdict, body = await self._call(
                    reviewer_backend,
                    self._reviewer_conv(conv, candidate_turn, exchange),
                    session_key, account_usage,
                )
            except Exception as exc:
                metrics["debate_error"] = str(exc)
                return self._finish(
                    DebateOutcome("reviewer_error", rounds, original_events if original_shipped else candidate_events),
                    metrics, start, session_key, parent_request_id,
                )
            round_record["verdict"] = verdict
            if verdict in ("approve", "flag"):
                if verdict == "flag":
                    round_record["objection"] = body[:600]
                self._log(round_record)
                return self._finish(
                    DebateOutcome("consensus", rounds, candidate_events),
                    metrics, start, session_key, parent_request_id,
                )
            if verdict == "malformed":
                round_record["raw_reply"] = body[:600]  # sensor: mine these to fix the protocol prompt
                self._log(round_record)
                metrics["debate_error"] = "malformed_verdict"
                return self._finish(
                    DebateOutcome("reviewer_error", rounds, candidate_events),
                    metrics, start, session_key, parent_request_id,
                )
            # verdict == objection
            objection = body or "unspecified objection"
            round_record["objection"] = objection[:600]
            fingerprint = _objection_hash(objection)
            if fingerprint in seen_objections:
                self._log(round_record)
                return self._finish(
                    DebateOutcome("deadlock", rounds, candidate_events, objection),
                    metrics, start, session_key, parent_request_id,
                )
            seen_objections.add(fingerprint)

            try:
                counter, counter_body = await self._call(
                    reviewer_backend,
                    self._counter_conv(conv, candidate_turn, objection),
                    session_key, account_usage,
                )
            except Exception as exc:
                metrics["debate_error"] = str(exc)
                self._log(round_record)
                return self._finish(
                    DebateOutcome("reviewer_error", rounds, candidate_events, objection),
                    metrics, start, session_key, parent_request_id,
                )
            if counter == "rebut" or counter == "malformed":
                # a malformed counter still reads as an argument; let the
                # reviewer judge it next round (no-progress valve backstops)
                round_record["counter"] = "rebut"
                round_record["rebuttal"] = counter_body[:600]
                self._log(round_record)
                exchange = (objection, counter_body or "(no rebuttal text)")
                continue
            # concede: regenerate through the caller's relay path
            round_record["counter"] = "concede"
            self._log(round_record)
            if regenerate is None:
                return self._finish(
                    DebateOutcome("deadlock", rounds, candidate_events, objection),
                    metrics, start, session_key, parent_request_id,
                )
            new_events = list(await regenerate(self._regen_conv(conv, candidate_turn, objection)))
            new_turn = _turn_from_events(new_events)
            if _candidate_hash(new_turn) == _candidate_hash(candidate_turn):
                return self._finish(
                    DebateOutcome("deadlock", rounds, candidate_events, objection),
                    metrics, start, session_key, parent_request_id,
                )
            if original_shipped:
                _account_discarded(new_events)  # regenerated candidates never ship
            else:
                _account_discarded(candidate_events)  # the replaced candidate never ships
            candidate_events, candidate_turn = new_events, new_turn
            exchange = None

    async def enforce_stream(
        self,
        relay_events,
        conv: Conversation,
        pool,
        metrics: dict,
        *,
        backend=None,
        regenerate=None,
        session_key: str | None = None,
        parent_request_id: str | None = None,
        account_usage=None,
    ):
        """Wrap the relay's event stream for enforce mode: buffer the
        candidate turn, debate it, ship the consensus events — emitting Ping
        keepalives whenever nothing else is flowing."""
        interval = self.cfg.keepalive_interval_s
        collected: list[IREvent] = []
        async for ev in keepalive_iter(relay_events, interval):
            if isinstance(ev, Ping):
                yield ev
                continue
            collected.append(ev)
        task = asyncio.ensure_future(self.run(
            conv, collected, pool, metrics,
            backend=backend, regenerate=regenerate,
            session_key=session_key, parent_request_id=parent_request_id,
            account_usage=account_usage,
        ))
        while not task.done():
            finished, _ = await asyncio.wait({task}, timeout=interval)
            if not finished:
                yield Ping()
        outcome = task.result()
        note = (
            outcome.outcome in ("deadlock", "deadline")
            and outcome.unresolved_objection
            and _is_prose_only(outcome.events)
        )
        for ev in outcome.events:
            if note and isinstance(ev, Done):
                yield TextDelta(
                    f"\n[harness] reviewer objection: {outcome.unresolved_objection[:300]}\n"
                )
                note = False
            yield ev

    # ---------- internals ----------

    def _reviewer_conv(
        self, conv: Conversation, candidate_turn: Turn, exchange: tuple[str, str] | None
    ) -> Conversation:
        turns = conv.turns + (candidate_turn,)
        if exchange is None:
            turns += (Turn("user", (TextPart(REVIEWER_INSTRUCTION),)),)
        else:
            objection, rebuttal = exchange
            turns += (
                Turn("user", (TextPart(f"OBJECTION:\n{objection}"),)),
                Turn("assistant", (TextPart(rebuttal),)),
                Turn("user", (TextPart(REVIEWER_REBUTTAL_INSTRUCTION),)),
            )
        return replace(conv, turns=turns)

    def _counter_conv(self, conv: Conversation, candidate_turn: Turn, objection: str) -> Conversation:
        turns = conv.turns + (
            candidate_turn,
            Turn("user", (TextPart(COUNTER_INSTRUCTION.format(objection=objection)),)),
        )
        return replace(conv, turns=turns)

    def _regen_conv(self, conv: Conversation, candidate_turn: Turn, objection: str) -> Conversation:
        turns = conv.turns + (
            candidate_turn,
            Turn("user", (TextPart(REGEN_FEEDBACK.format(objection=objection)),)),
        )
        return replace(conv, turns=turns)

    async def _call(
        self, reviewer_backend, debate_conv: Conversation,
        session_key: str | None, account_usage,
    ) -> tuple[str, str]:
        # No debate-specific output ceiling: the reviewer inherits the same
        # max_tokens the client requested for the turn (owner decision
        # 2026-07-19 — the harness invents no token limits for the debate).
        payload = reviewer_backend.profile.render(debate_conv, reviewer_backend.model_name)
        text = ""
        done = Done("end_turn")
        ttft_ms: int | None = None
        start = time.monotonic()
        async for ev in reviewer_backend.profile.parse(reviewer_backend.stream(payload)):
            if ttft_ms is None:
                ttft_ms = int((time.monotonic() - start) * 1000)
            if isinstance(ev, TextDelta):
                text += ev.text
            elif isinstance(ev, Done):
                done = ev
        if account_usage:
            account_usage(reviewer_backend, done, session_key, count_request=True, ttft_ms=ttft_ms)
        return _parse_verdict(text)

    def _log(self, record: dict) -> None:
        if self.reviews_logger:
            self.reviews_logger.write(record)

    def _finish(
        self, outcome: DebateOutcome, metrics: dict, start: float,
        session_key: str | None, parent_request_id: str | None,
    ) -> DebateOutcome:
        wall_ms = int((time.monotonic() - start) * 1000)
        metrics["debate_outcome"] = outcome.outcome
        metrics["debate_rounds"] = outcome.rounds
        metrics["debate_wall_ms"] = wall_ms
        metrics["debate_unresolved"] = bool(outcome.unresolved_objection)
        if self.reviews_logger:
            self.reviews_logger.write({
                "kind": "debate",
                "outcome": outcome.outcome,
                "rounds": outcome.rounds,
                "wall_ms": wall_ms,
                "unresolved_objection": (outcome.unresolved_objection or "")[:600] or None,
                "session_key": session_key,
                "parent_request_id": parent_request_id,
            })
        return outcome
