"""The agentic coaching loop — Koutsi pulls what it needs (issue #43).

Everywhere else in this codebase an LLM is handed a fixed context blob assembled
ahead of time: ``_build_status_prompt`` is ~145 lines of string building that
runs in full whether the answer needs all of it or not, and cannot follow a
thread — if the form number looks off, it has no way to go and look at the last
three weeks of rides to say *why*. This module inverts that. It hands the model
the tool layer from issue #42 and lets it ask.

What this is and is not
-----------------------
It is **not** a chat endpoint. Both surfaces it serves — the dashboard's daily
training-status card and per-activity analysis — are one-shot generations fired
from a background task with nobody typing, so there is no conversation to store
and no conversation id to hand out. The message history exists inside one
``analyze_*_bg`` task and dies with it.

It is **not** an SSE stream either. openkoutsi persists generations to the DB and
polls, deliberately: a local model may take minutes, and a page reload must not
lose a run. So the loop is layered *onto* :func:`stream_into_db` rather than
beside it — many turns in, one growing column out.

The silent gap
--------------
That polling model is exactly what makes an agent loop awkward here. The first N
round trips emit no assistant prose at all, just tool calls, so a card that used
to fill in token by token would instead show a spinner for a long time and then
dump a finished answer — worse than what it replaced. The loop therefore
persists **progress** on the same cadence as text, as codes from a fixed
vocabulary (:data:`PROGRESS_THINKING`, ``tool.<name>``) that the web app
localises. Codes rather than model-authored sentences because the prompts run in
fourteen languages while every tool name and description is English, and because
a code cannot leak tool internals into the card.

Degrading, not failing
----------------------
BYOK is what makes this harder here than in a product that owns its model. Users
point openkoutsi at whatever OpenAI-compatible server they like, and
tool-calling support across that population ranges from good, to absent, to
present-but-wrong. "Smoke-test each provider" is not available to the hoster, so
every failure mode degrades at runtime, per call, to the blob prompt:

============================================  ===========================================
The provider…                                 …and the run
============================================  ===========================================
rejects the ``tools`` param (400/422)         falls back, via ``is_tool_calling_unsupported_error``
is flagged ``"tools_supported": false``       never tried
accepts ``tools`` and calls none              falls back — an answer built from no data
emits malformed tool-call JSON                the error becomes a tool result; the run continues
keeps calling tools past a budget             one forced final turn, then falls back
fails any other way before prose is written   falls back — 429, 5xx, a dropped connection,
                                              and above all a context-length 400, which is a
                                              failure *this loop creates*
says our own function schema is invalid       **raises** — that is our bug, not their limit
============================================  ===========================================

:exc:`AgenticUnavailable` is the one signal for all of them, and it carries a
hard rule: it may only be raised *before the first character of prose has been
yielded*. Once text is out it has been committed to the DB, and a fallback would
staple a second answer onto the first. :func:`agentic_stream` enforces that
rather than trusting it — and that invariant is exactly what makes the broad
"any other failure" rule safe.

Two budgets stop the gathering, and both route to the same forced final turn:
the round cap (:data:`MAX_ROUNDS_STATUS` / :data:`MAX_ROUNDS_ACTIVITY`) and the
run's total tool-result size (:data:`MAX_RUN_RESULT_CHARS`). The second exists
because the first bounds the wrong quantity — a round may carry any number of
parallel calls, and it is their *sum*, replayed into every later turn, that
spends the context window and the money. :data:`MAX_CALLS_PER_TURN` bounds the
breadth of a single turn for the same reason.

The blob builders are not legacy
--------------------------------
``_build_status_prompt`` and ``_build_prompt`` stay, stay tested, and stay the
answer for every provider that cannot do this, plus the bulk-import paths where
five times the calls is a real bill and nobody reads the output one by one. Plan
for both paths to coexist indefinitely.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

import httpx

from ..core.config import settings
from ..mcp.dispatch import ToolCaller, ToolResult, call_tool
from ..mcp.registry import Tool, all_tools
from ..models.user_orm import Athlete
from .llm_access import merge_usage
from .llm_client import (
    ResolvedLlm,
    is_our_tool_schema_error,
    is_tool_calling_unsupported_error,
)
from .llm_streaming import (
    AgentProgress,
    StreamItem,
    StreamSetup,
    TextDelta,
    ToolCallDelta,
    resolve_stream_setup,
    stream_completion_events,
)

log = logging.getLogger(__name__)

#: A single-shot stream factory — the existing blob path, handed its own usage
#: dict exactly as :func:`stream_into_db` hands one to any stream.
BlobStream = Callable[[dict], AsyncIterator[str]]


# ── Progress vocabulary ─────────────────────────────────────────────────────
#
# Every value here is a key the web app translates. Two rules keep it a
# contract rather than a leak: it is closed at the top level (`thinking`,
# `tool.*`), and a client that meets a `tool.*` code it does not know must fall
# back to its generic copy. That is what lets a tool be added in #42 without
# shipping a frontend release in lockstep.

#: The model is deciding what to look at, before it has asked for anything. Once
#: a tool has been called the code stays on *that tool* through the turn that
#: reads its result, because that turn is the slow part and naming what it is
#: working on is the whole point.
PROGRESS_THINKING = "thinking"

#: Prefix for "Koutsi is calling <tool>". The suffix is the registry tool name.
PROGRESS_TOOL_PREFIX = "tool."


def progress_code_for_tool(tool_name: str) -> str:
    """``"get_power_profile"`` → ``"tool.get_power_profile"``."""
    return f"{PROGRESS_TOOL_PREFIX}{tool_name}"


def progress_vocabulary() -> list[str]:
    """Every progress code this build can emit, for tests and for the docs.

    Derived from the registry rather than restated, so a tool added to #42's
    layer cannot quietly acquire a code nobody translated — the frontend parity
    test reads this list.
    """
    return [PROGRESS_THINKING] + [progress_code_for_tool(t.name) for t in all_tools()]


# ── Budgets ─────────────────────────────────────────────────────────────────

#: Tool-calling rounds allowed before the loop forces an answer. The two
#: surfaces genuinely differ: the status card is a broad question that wants
#: several lookups (load trend, then the plan, then maybe the power curve to
#: explain a flat form number), while a single activity is narrow and normally
#: needs one or two — its own detail, occasionally something to compare against.
#: Cheap to raise, expensive to have wrong: every round adds a completion *and*
#: carries every previous tool result in its context.
MAX_ROUNDS_STATUS = 6
MAX_ROUNDS_ACTIVITY = 3

#: Tool calls dispatched from any single turn. The round cap bounds *round
#: trips*, and a round trip may carry any number of parallel calls — so without
#: this the worst case is not six calls but six times however many the model
#: emits at once. A model that shotguns the whole registry (the failure the
#: ``llm-eval`` agentic family exists to detect) would otherwise run 54 calls and
#: replay every result into every later turn, each of those turns billed.
#: Anything past this gets a result explaining it was not run, so the model can
#: ask again rather than reason from a gap it does not know about.
MAX_CALLS_PER_TURN = 4

#: Total tool-result characters one run may accumulate. ``MAX_TOOL_RESULT_CHARS``
#: bounds a single result; nothing bounded their sum, and the sum is what is
#: replayed into the context of every subsequent turn. Once this is crossed the
#: loop stops offering tools and goes to the forced final turn — the path that
#: already exists for the round cap, reached by a different trigger. It is also
#: the cheapest defence against blowing a small context window, which is a real
#: risk on the self-hosted models BYOK points at.
MAX_RUN_RESULT_CHARS = 24_000

#: One tool call may not exceed this. The tools are aggregate reads over one
#: user's SQLite file, so anything approaching it is a pathological query rather
#: than a slow one, and a run must not be able to sit on the pending status until
#: the 30-minute timeout because a single call hung.
TOOL_TIMEOUT_S = 30.0

#: A tool result longer than this is truncated before it enters the context,
#: with the marker below. `call_tool` already refuses anything over 64 KiB as a
#: shaping bug; this is the much tighter bound that matters to a model's context
#: window, and it is a *truncation* rather than a refusal so the model still sees
#: the head of the answer and can narrow its next query.
MAX_TOOL_RESULT_CHARS = 6000
TRUNCATION_MARKER = (
    "\n\n[… truncated: {omitted} more characters. You are NOT seeing the whole "
    "result. Narrow the request — a shorter date range, a smaller 'limit' — if "
    "you need the rest.]"
)

# ── Why a run could not be served agentically ───────────────────────────────
#
# Carried on :class:`AgenticUnavailable`. Surfaces with a blob prompt ignore
# these entirely; chat renders one localised sentence per code.

#: No more specific cause — the default.
CODE_UNAVAILABLE = "unavailable"
#: Every agent slot was taken, and stayed taken for as long as we waited.
CODE_BUSY = "busy"
#: This model cannot call tools: flagged ``tools_supported=false``, or it
#: rejected the ``tools`` param outright. A settled property of the provider
#: rather than a transient failure, so the web app disables chat on it rather
#: than inviting a retry that will fail the same way.
CODE_TOOLS_UNSUPPORTED = "tools_unsupported"
#: The provider answered, but with neither tool calls nor prose.
CODE_NO_ANSWER = "no_answer"
#: A 429, a 5xx, or a context-length 400 — worth retrying.
CODE_UPSTREAM = "upstream"
#: The provider could not be reached at all: reset connection, timeout, DNS.
CODE_UNREACHABLE = "unreachable"


#: Runs in flight in this process. See ``Settings.agent_max_concurrent_runs``.
#:
#: A plain counter rather than an :class:`asyncio.Semaphore`, because the one
#: thing this guard must never do is *wait* — and waiting is a semaphore's whole
#: purpose. Testing ``.locked()`` before ``async with`` looks non-blocking and is
#: not: between the check and the acquire, another run can take the last slot,
#: and this one then blocks on exactly the acquire it meant to skip. A counter
#: closes that gap by construction — asyncio is cooperative, there is no
#: ``await`` between reading it and incrementing it, so the check and the claim
#: are one indivisible step.
_active_runs = 0


def _try_claim_slot() -> bool:
    """Take a slot if one is free. Never waits, never raises.

    The check and the claim are one indivisible step by construction: asyncio is
    cooperative and there is no ``await`` between reading the counter and
    incrementing it, which is the property the counter exists for (see
    :data:`_active_runs`). Both acquisition policies below are built on this, so
    neither can reintroduce the check-then-acquire gap.
    """
    global _active_runs
    if _active_runs >= max(1, int(settings.agent_max_concurrent_runs)):
        return False
    _active_runs += 1
    return True


def _release_slot() -> None:
    global _active_runs
    _active_runs -= 1


@contextmanager
def _run_slot() -> Iterator[None]:
    """Claim a slot for one background run, or refuse immediately.

    Refusing is the design. Waiting would push the run towards the 30-minute
    pending timeout with the athlete watching a spinner, and the blob prompt is
    a worse answer available *now* — the better trade under load.
    """
    if not _try_claim_slot():
        limit = max(1, int(settings.agent_max_concurrent_runs))
        raise AgenticUnavailable(
            f"all {limit} agent slots are in use", code=CODE_BUSY
        )
    try:
        yield
    finally:
        _release_slot()


#: How often a queued interactive turn re-checks for a free slot.
_SLOT_POLL_INTERVAL_S = 0.25


@asynccontextmanager
async def _waited_run_slot(timeout_s: float) -> AsyncIterator[None]:
    """Claim a slot for an interactive run, waiting up to ``timeout_s`` (#44).

    The opposite trade from :func:`_run_slot`, for the opposite situation. That
    one refuses instantly because a *better answer exists* — the blob prompt —
    and waiting would only delay it. Chat has no blob prompt, so refusing
    instantly buys nothing and costs the athlete their question: the four slots
    are shared with the background training-status runs that fire on dashboard
    load, and "ask a question just after opening the dashboard" is an ordinary
    thing to do, not an edge case.

    So an interactive turn waits, and the wait is *visible* — the assistant row
    sits in ``queued`` until the slot is claimed, which is a state the athlete
    can read rather than a spinner that means nothing. The bound is what keeps
    this from becoming the failure mode :func:`_run_slot` was written to avoid.

    Polling rather than an :class:`asyncio.Event` signalled on release: the
    counter is a plain module global shared by whatever loops this process runs,
    and an Event is bound to the loop that created it. A quarter-second poll on
    a path that is idle unless the instance is saturated is not worth the
    loop-affinity bug that would buy. Waiters are not FIFO — under contention
    whichever turn next wakes wins — which on a single-athlete instance with
    four slots is a distinction without a difference.
    """
    deadline = time.monotonic() + max(0.0, timeout_s)
    while not _try_claim_slot():
        if time.monotonic() >= deadline:
            limit = max(1, int(settings.agent_max_concurrent_runs))
            raise AgenticUnavailable(
                f"all {limit} agent slots were still in use after "
                f"{timeout_s:.0f}s",
                code=CODE_BUSY,
            )
        await asyncio.sleep(_SLOT_POLL_INTERVAL_S)
    try:
        yield
    finally:
        _release_slot()


class AgenticUnavailable(Exception):
    """This run cannot be served agentically; use the blob prompt instead.

    Never surfaced to the athlete **on the surfaces that have a blob prompt**.
    For them it means "take the other path", and every raise site is a runtime
    discovery about the provider or the load on this process, not an error in
    the request.

    Chat (issue #44) has no other path — the question is arbitrary, so there is
    no pre-assembled context to fall back to — which is what ``code`` is for.
    There it is the difference between five quite different causes and one
    generic apology: "Koutsi is busy finishing your daily check-in, try again in
    a moment" is a true and actionable sentence, and "something went wrong" is
    neither. A machine key rather than the message, because the message is a log
    line and the athlete reads fourteen languages.
    """

    def __init__(self, message: str, *, code: str = CODE_UNAVAILABLE) -> None:
        super().__init__(message)
        self.code = code


# ── The OpenAI chat-completions tool dialect ────────────────────────────────
#
# Everything provider-specific is behind this seam. The loop below deals in
# `PendingToolCall` and registry `Tool`s; only these two functions know that a
# tool definition is `{"type": "function", "function": {...}}` and that a result
# goes back as a `role: "tool"` message keyed by `tool_call_id`.


def tool_definitions(tools: list[Tool]) -> list[dict]:
    """Render registry tools into the OpenAI ``tools`` array.

    The schema is the tool's own pydantic arguments model, so what the provider
    constrains the model to and what
    :func:`~backend.app.mcp.dispatch.call_tool` validates against are the same
    object and cannot drift. Names already satisfy the dialect's
    ``^[a-zA-Z0-9_-]{1,64}$`` by registry rule, so nothing is mapped at this
    boundary — a mapping is a place for a name to get lost.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema(),
            },
        }
        for tool in tools
    ]


def _assistant_message(text: str, calls: list["PendingToolCall"]) -> dict:
    """The assistant turn to replay, carrying the calls it made."""
    message: dict = {"role": "assistant", "content": text or None}
    message["tool_calls"] = [
        {
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": call.raw_arguments},
        }
        for call in calls
    ]
    return message


def _tool_message(call: "PendingToolCall", content: str) -> dict:
    return {"role": "tool", "tool_call_id": call.id, "content": content}


# ── Assembling streamed tool calls ──────────────────────────────────────────


@dataclass
class PendingToolCall:
    """One tool call, reassembled from its deltas."""

    id: str
    name: str
    raw_arguments: str = ""

    def parse_arguments(self) -> tuple[Optional[dict], Optional[str]]:
        """``(arguments, error)`` — never raises.

        A model that emits broken JSON must get a sentence back and another
        turn, not a 500. Two shapes are wrong in different ways and both happen:
        unparseable text, and valid JSON that is not an object (``"[]"``,
        ``"null"``, a bare string). Empty arguments are legitimate — several
        tools take none — so they parse to ``{}``.
        """
        raw = (self.raw_arguments or "").strip()
        if not raw:
            return {}, None
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return None, (
                f"The arguments to '{self.name}' were not valid JSON, so the tool "
                f"was not called. Send the arguments as a JSON object, for "
                f"example {{}} for no arguments. What arrived was: {raw[:200]}"
            )
        if not isinstance(parsed, dict):
            return None, (
                f"The arguments to '{self.name}' must be a JSON object, but a "
                f"{type(parsed).__name__} arrived. Send an object keyed by "
                f"argument name, for example {{}} for no arguments."
            )
        return parsed, None


class _CallAssembler:
    """Collects :class:`ToolCallDelta` fragments into whole calls.

    Keyed on the delta's ``index`` because that is the only field every provider
    sends on every fragment — ``id`` and ``name`` normally arrive once, on the
    first.
    """

    def __init__(self) -> None:
        self._by_index: dict[int, dict[str, str]] = {}
        self._order: list[int] = []

    def add(self, delta: ToolCallDelta) -> None:
        slot = self._by_index.get(delta.index)
        if slot is None:
            slot = {"id": "", "name": "", "arguments": ""}
            self._by_index[delta.index] = slot
            self._order.append(delta.index)
        if delta.id:
            slot["id"] = delta.id
        if delta.name:
            # Appended, not replaced: a few servers stream the function name in
            # fragments the way they stream arguments.
            slot["name"] += delta.name
        if delta.arguments:
            slot["arguments"] += delta.arguments

    @property
    def started(self) -> bool:
        return bool(self._by_index)

    def finish(self) -> list[PendingToolCall]:
        """The assembled calls, with ids made unique and unnamed ones dropped.

        Two defences against a provider emitting function calling badly:

        * **A call with no name is dropped.** There is nothing to dispatch and
          no way to describe the failure back to the model in a way it could act
          on — it never learns which of its calls vanished.
        * **Duplicate ids are made unique.** The dialect requires exactly one
          ``role: "tool"`` message per ``tool_call_id``, and a repeated id would
          make that impossible to satisfy: two results under one id is a 400 on
          the next request, and one result for two calls is a different 400. A
          synthesised suffix keeps the pairing one-to-one, which is the property
          the provider is actually checking.
        """
        calls: list[PendingToolCall] = []
        seen: set[str] = set()
        for position, index in enumerate(self._order):
            slot = self._by_index[index]
            name = slot["name"].strip()
            if not name:
                log.warning("agent: dropping a tool call with no function name")
                continue
            call_id = slot["id"].strip() or f"call_{position}"
            if call_id in seen:
                call_id = f"{call_id}_{position}"
            seen.add(call_id)
            calls.append(
                PendingToolCall(id=call_id, name=name, raw_arguments=slot["arguments"])
            )
        return calls


# ── One turn ────────────────────────────────────────────────────────────────


@dataclass
class _Turn:
    """What one completion produced, once it has finished."""

    text: str = ""
    calls: list[PendingToolCall] = field(default_factory=list)
    #: Text already yielded downstream — and therefore already in the DB.
    emitted: bool = False


#: How much prose to hold back before deciding a turn is the final answer.
#:
#: A turn that is going to call a tool sometimes narrates first ("Let me look at
#: your last four weeks…"), and that preamble must never reach the card: the
#: column it lands in is the answer. So content is buffered until either a
#: tool-call delta arrives — proving the turn is a tool round, and the buffer is
#: discarded — or this many characters accumulate with no tool call in sight, at
#: which point the turn is committed to being the answer and everything after it
#: streams live. Small enough that the card still fills in visibly; large enough
#: to clear any realistic preamble.
_COMMIT_AFTER_CHARS = 240


# ── The loop ────────────────────────────────────────────────────────────────


@dataclass
class AgentRequest:
    """Everything one agentic run needs.

    Deliberately **not** the caller's session. The tools open their own, one per
    call — see :func:`_dispatch` for why that is worth a connection checkout.
    ``athlete`` is here only to resolve the athlete's LLM config and opt-in; the
    copy the tools read is loaded inside their own session from ``user_id``,
    which is what every check and every audit record keys on anyway.
    """

    athlete: Athlete
    user_id: str
    system_prompt: str
    user_prompt: str
    feature: str
    max_rounds: int
    #: Restated verbatim on the final turn — see :func:`_final_reminder`.
    format_rule: str
    #: The calendar date the tools reckon from, in the **athlete's** timezone.
    #: Six of the nine key off it, and they are the date-boundary-sensitive ones.
    #: The server's own date is the wrong answer for anyone far enough from UTC,
    #: and the brief already tells the model what day it is — the two must agree.
    today: date = field(default_factory=date.today)
    tools: Optional[list[Tool]] = None

    # ── Conversational runs (issue #44) ─────────────────────────────────────

    #: Prior turns, oldest first, as ``{"role": …, "content": …}``. When set it
    #: replaces ``user_prompt`` as the loop's starting history, with the new
    #: question already its last entry — trimming and system-prompt assembly
    #: happen in :mod:`.llm_chat`, which owns what a conversation *is*, leaving
    #: this loop the one job it already had.
    history: Optional[list[dict]] = None

    #: Is a human waiting on this run? Three behaviours turn on it, and all
    #: three are cases where the right answer for a card is the wrong one for a
    #: conversation:
    #:
    #: * the first turn's prose may be the answer. The card suppresses it,
    #:   correctly — an answer written before looking at anything is guesswork
    #:   about *this athlete's training*. But "what does TSB actually mean?" is
    #:   a coaching question with no lookup behind it, and refusing to answer it
    #:   without calling a tool first would be a worse conversation, not a
    #:   safer one;
    #: * a slot is waited for rather than refused (:func:`_waited_run_slot`);
    #: * failures carry a ``code`` the athlete sees, since there is no blob
    #:   prompt to quietly serve instead.
    conversational: bool = False

    #: How long an interactive turn may sit queued before giving up.
    slot_wait_s: float = 0.0


def _final_reminder(format_rule: str) -> dict:
    """The format rule, restated as a system message on the answering turn.

    Models are measurably worse at obeying a leading-format instruction on a turn
    that follows tool results than on a clean single-shot prompt, and the
    ``MOOD:`` line is not decoration: ``parseMoodAndParagraphs`` reads it to pick
    Koutsi's avatar, and the ``llm-eval`` harness asserts on it. Restating it
    where the model is about to answer costs a few tokens and is the difference
    between the contract holding and holding usually.
    """
    return {
        # A *user* turn, not a system one, and that is the whole point of the
        # placement. Several chat templates in the llama.cpp / Ollama family
        # render only the leading system message and silently drop later ones —
        # so a mid-conversation system reminder would be a no-op exactly on the
        # models most likely to need it, with nothing in the logs to say so.
        # Every template renders a user turn. The instance house style keeps its
        # system role because it sits at the front, where those templates still
        # pick it up.
        "role": "user",
        "content": (
            "You now have everything you asked for. Write the athlete's feedback "
            "as your next message, using only the tool results above — do not "
            "call any more tools.\n\n" + format_rule
        ),
    }


def agentic_enabled(athlete: Any) -> bool:
    """Has this athlete opted into the agentic coach? (issue #43, open question 2)

    Opt-in rather than "on whenever the model supports tools", for the length of
    time it takes to trust it. The blob prompts are well-tuned after several
    rounds of prompt work, and the claimed win here — following a thread — is
    the kind of thing that either shows up in the output or doesn't. An opt-in
    makes that comparison honest and makes the rollback free: a toggle, not a
    deploy.
    """
    settings_dict = (getattr(athlete, "app_settings", None) or {}) if athlete is not None else {}
    return bool(settings_dict.get("agentic_koutsi"))


async def coaching_stream(
    *,
    request: Optional[AgentRequest],
    blob: "BlobStream",
    usage_out: dict,
) -> AsyncIterator[StreamItem]:
    """One stream for a coaching surface: agentic when it can be, blob when not.

    Both analyzers call exactly this, so the fallback policy — and the token
    arithmetic that goes with it — is written and tested once rather than twice
    with a subtle difference. ``request`` is ``None`` when the surface is not
    even attempting the agentic path (the athlete has not opted in, or this is a
    bulk import); ``blob`` is the single-shot stream factory, called with its own
    usage dict.

    Whatever happens, ``usage_out`` ends up holding the **whole run's** cost. A
    fallback after two tool rounds spent real tokens on those rounds, and the
    hoster paid for them; dropping them because the answer eventually came from
    the other path would under-report exactly the runs that cost the most.
    """
    agent_usage: dict = {}
    blob_usage: dict = {}
    try:
        if request is not None:
            try:
                async for item in agentic_stream(request, agent_usage):
                    yield item
                return
            except AgenticUnavailable as exc:
                log.info(
                    "agent: %s (feature=%s) — using the single-shot prompt",
                    exc,
                    request.feature,
                )
                # Whatever the tool rounds left on the card is now a lie about
                # what is happening. Clear it before the prose starts arriving
                # from somewhere else entirely.
                yield AgentProgress(None)

        async for chunk in blob(blob_usage):
            yield chunk
    finally:
        cfg = blob_usage.get("cfg") or agent_usage.get("cfg")
        if cfg is not None:
            usage_out["cfg"] = cfg
        merged = merge_usage(agent_usage.get("usage"), blob_usage.get("usage"))
        if merged is not None:
            usage_out["usage"] = merged


def _format_reminder(format_rule: str) -> dict:
    """The format rule alone, for a turn that follows tool results.

    Distinct from :func:`_final_reminder`, which also forbids further tool
    calls: sending *that* on every turn would end the loop after round one. A
    user turn for the same reason as the final reminder — a mid-conversation
    system message is dropped outright by some chat templates.
    """
    return {"role": "user", "content": format_rule}


def _budget_exhausted(rounds: int, spent: int, max_rounds: int) -> Optional[str]:
    """Which budget, if any, says stop gathering — named for the log line."""
    if rounds >= max_rounds:
        return f"the {max_rounds}-round cap"
    if spent >= MAX_RUN_RESULT_CHARS:
        return f"the {MAX_RUN_RESULT_CHARS}-character tool-result budget"
    return None


def _bounded(calls: list[PendingToolCall]) -> tuple[list[PendingToolCall], list[PendingToolCall]]:
    """Split a turn's calls into the ones to run and the ones to refuse."""
    return calls[:MAX_CALLS_PER_TURN], calls[MAX_CALLS_PER_TURN:]


async def agentic_stream(
    request: AgentRequest, usage_out: dict
) -> AsyncIterator[StreamItem]:
    """Run the loop, yielding progress codes and then the answer's prose.

    Drained by :func:`~backend.app.services.llm_streaming.stream_into_db` exactly
    like a single-shot stream, so the 500 ms commit cadence, the status settling
    and the usage recording are all unchanged.

    Raises :exc:`AgenticUnavailable` when the run should be served by the blob
    prompt instead — but only ever before the first character of prose has been
    yielded, since after that the fallback would append a second answer to a
    partially-written first one. The check is enforced below, not assumed.
    """
    emitted_any_text = False
    try:
        async for item in _run(request, usage_out):
            if isinstance(item, str) and item:
                emitted_any_text = True
            yield item
    except AgenticUnavailable:
        if emitted_any_text:
            # Not recoverable: prose is already committed. Let the drain loop
            # mark the failure rather than stapling a blob answer onto it.
            log.error(
                "agent: fallback requested after prose had already been written "
                "(feature=%s) — failing the run instead",
                request.feature,
            )
            raise RuntimeError(
                "The agentic run asked to fall back after it had already "
                "produced output."
            ) from None
        raise


@asynccontextmanager
async def _slot_for(request: AgentRequest) -> AsyncIterator[None]:
    """The acquisition policy this run's caller is entitled to.

    A background run refuses instantly and falls back; an interactive one waits,
    because it has nothing to fall back to. See :func:`_waited_run_slot`.
    """
    if request.conversational and request.slot_wait_s > 0:
        async with _waited_run_slot(request.slot_wait_s):
            yield
        return
    # Claimed before the first `await`, so a burst of runs starting together
    # cannot all see a free slot and then all take it.
    with _run_slot():
        yield


async def _run(request: AgentRequest, usage_out: dict) -> AsyncIterator[StreamItem]:
    async with _slot_for(request):
        setup = await resolve_stream_setup(
            request.athlete, request.user_id, usage_out=usage_out
        )
        if not setup.cfg.tools_supported:
            raise AgenticUnavailable(
                f"preset for model {setup.cfg.model!r} is flagged tools_supported=false",
                code=CODE_TOOLS_UNSUPPORTED,
            )

        async for item in _drive(request, setup, usage_out):
            yield item


async def _drive(
    request: AgentRequest,
    setup: StreamSetup,
    run_usage: dict,
) -> AsyncIterator[StreamItem]:
    """The conversation itself: turn, dispatch, turn, …, answer."""
    tools = request.tools if request.tools is not None else all_tools()
    definitions = tool_definitions(tools)
    caller = ToolCaller.internal(request.user_id)

    # Rebuilt on every turn rather than mutated in place, so the instance's
    # house style (`llm_analysis_context`) is a system message on turn five as
    # much as on turn one — the hoster's rules are not something three tool
    # results are allowed to push out of the model's attention.
    #
    # A conversational run starts from the stored dialogue (issue #44) with the
    # new question already at the end of it; every other run starts from the one
    # prompt its surface built. Copied either way: the tool rounds below append
    # to this list, and those appends must not reach the caller's list — that is
    # the trimmed *wire* history, and writing tool traffic back into it is how
    # the loop's scratch work would end up in the stored transcript.
    history: list[dict] = (
        list(request.history)
        if request.history is not None
        else [{"role": "user", "content": request.user_prompt}]
    )

    yield AgentProgress(PROGRESS_THINKING)

    rounds = 0
    spent = 0
    while True:
        forced = _budget_exhausted(rounds, spent, request.max_rounds)
        if forced is not None:
            break

        messages = setup.system_messages(request.system_prompt) + history
        if rounds > 0:
            # Restated on *every* turn that follows tool results, not only the
            # forced one. The reason `_final_reminder` gives — models obey a
            # leading-format instruction less reliably after tool results —
            # applies to whichever turn ends up answering, and the common shape
            # is answering after one or two rounds, not hitting the cap. Only
            # the format half: the forced reminder also says "call no more
            # tools", which here would end the loop after round one.
            messages = messages + [_format_reminder(request.format_rule)]
        turn = _Turn()
        async for item in _collect(
            _stream_turn(setup.cfg, messages, definitions, run_usage),
            turn,
            # Turn zero's prose is never the answer *on a card*: it would be one
            # written without having looked at anything, since the agentic
            # prompt carries no data of its own. Collect it for the replayed
            # assistant message, but never emit it.
            #
            # A conversation is the exception (issue #44). Not every question
            # has a lookup behind it — "what does TSB actually mean?", "why do
            # you keep saying my form is negative?" — and there the answer
            # written straight off is the right one. The preamble guard below
            # still applies: prose that turns out to precede a tool call is
            # discarded, so "let me look at your last four weeks…" never lands
            # in the answer even here.
            allow_text=rounds > 0 or request.conversational,
        ):
            yield item

        if not turn.calls:
            if turn.emitted:
                # A turn that answered after at least one tool round. Done.
                yield AgentProgress(None)
                return
            raise AgenticUnavailable(
                "the provider accepted 'tools' but called none"
                if rounds == 0
                else "the provider returned neither tool calls nor an answer",
                # On a card, "called no tools" means an answer built from no
                # data, and the blob prompt is strictly better. A conversational
                # run never reaches here for that reason — its turn-zero prose
                # is allowed, so answering without tools sets `emitted` and
                # returns above. Getting here at all means the provider produced
                # nothing whatsoever.
                code=CODE_NO_ANSWER if request.conversational else CODE_UNAVAILABLE,
            )

        if turn.emitted:
            # Text streamed *and* tool calls, on a turn we had already committed
            # to being the answer. The text is in the DB and cannot be unsaid,
            # so the calls are dropped and the run ends here — a complete answer
            # that ignored a late tool call beats a truncated one.
            log.info(
                "agent: ignoring %d tool call(s) requested after the answer had "
                "started (feature=%s)",
                len(turn.calls),
                request.feature,
            )
            yield AgentProgress(None)
            return

        run, dropped = _bounded(turn.calls)
        history.append(_assistant_message(turn.text, turn.calls))
        for call in run:
            yield AgentProgress(progress_code_for_tool(call.name))
            content = await _dispatch(request, caller, call)
            spent += len(content)
            history.append(_tool_message(call, content))
        for call in dropped:
            # Still exactly one result per call — the pairing the dialect
            # checks — but a sentence saying why, so the model knows it is
            # missing something rather than quietly reasoning without it.
            history.append(
                _tool_message(
                    call,
                    f"'{call.name}' was not run: this turn asked for "
                    f"{len(turn.calls)} tools at once and only the first "
                    f"{MAX_CALLS_PER_TURN} were executed. Ask again on the next "
                    "turn for whatever you still need, a few at a time.",
                )
            )
            _log_call(request, call, "dropped_over_breadth", 0.0, None)
        rounds += 1
        # Deliberately *not* reset to `thinking` here. A tool call against one
        # user's SQLite file takes milliseconds; the model turn that reads the
        # result takes seconds. Resetting would leave the card on a generic
        # "thinking" for all of the slow part and flash the informative line for
        # none of it — the exact inversion of what the progress line is for.
        # "Koutsi is checking your power curve…" should stay up while Koutsi is,
        # in fact, working out what the power curve means.

    # A budget ran out. One more turn with no tools offered at all — not merely
    # `tool_choice: "none"`, which some servers ignore — so a model stuck in a
    # calling loop has nothing left to call.
    log.info(
        "agent: hit %s (feature=%s), forcing a final answer", forced, request.feature
    )
    messages = (
        setup.system_messages(request.system_prompt)
        + history
        + [_final_reminder(request.format_rule)]
    )
    final = _Turn()
    async for item in _collect(
        _stream_turn(setup.cfg, messages, None, run_usage), final, allow_text=True
    ):
        yield item
    if not final.emitted:
        raise AgenticUnavailable(
            "the forced final turn produced no answer", code=CODE_NO_ANSWER
        )
    yield AgentProgress(None)


async def _collect(
    events: AsyncIterator[Any],
    turn: _Turn,
    *,
    allow_text: bool,
) -> AsyncIterator[StreamItem]:
    """Drain one turn's events, yielding prose only once it is safe to.

    See :data:`_COMMIT_AFTER_CHARS` for why the first characters are held back.
    ``allow_text=False`` suppresses prose entirely — turn zero, whose only
    possible answer is one written without having looked at anything.
    """
    assembler = _CallAssembler()
    buffered: list[str] = []
    committed = False

    async for event in events:
        if isinstance(event, ToolCallDelta):
            assembler.add(event)
            if not committed:
                # A tool round after all: whatever prose came first was a
                # preamble, and the answer column is not where preambles go.
                buffered.clear()
            continue
        if not isinstance(event, TextDelta) or not event.text:
            continue
        turn.text += event.text
        if not allow_text or assembler.started:
            continue
        if committed:
            yield event.text
            continue
        buffered.append(event.text)
        if sum(len(part) for part in buffered) >= _COMMIT_AFTER_CHARS:
            committed = True
            turn.emitted = True
            yield "".join(buffered)
            buffered.clear()

    turn.calls = assembler.finish()
    if buffered and allow_text and not turn.calls:
        # A short answer that never reached the commit threshold. Safe now: the
        # turn is over and no tool call arrived.
        turn.emitted = True
        yield "".join(buffered)


async def _stream_turn(
    cfg: ResolvedLlm,
    messages: list[dict],
    definitions: Optional[list[dict]],
    run_usage: dict,
) -> AsyncIterator[Any]:
    """One completion, with this turn's usage folded into the run's total.

    Every turn is billed separately by the provider, so the run's cost is the sum
    and recording only the last would under-report an analysis by however many
    turns it took (issue #43's concrete accounting bug). The fold happens in a
    ``finally`` so a turn that failed halfway still contributes whatever the
    provider had already reported.
    """
    turn_usage: dict = {}
    try:
        async for event in stream_completion_events(
            cfg, messages, tools=definitions, usage_out=turn_usage
        ):
            yield event
    except httpx.HTTPStatusError as exc:
        if definitions and is_tool_calling_unsupported_error(exc):
            log.warning(
                "LLM tool calling unsupported: provider rejected 'tools' "
                "(model=%s) — falling back to the single-shot prompt",
                cfg.model,
            )
            raise AgenticUnavailable(
                "provider rejected the 'tools' param", code=CODE_TOOLS_UNSUPPORTED
            ) from exc
        if definitions and is_our_tool_schema_error(exc):
            # Our own pydantic model is broken. Degrading here would hide it
            # behind a quietly worse answer for every athlete on every provider.
            raise
        # Anything else upstream — a 429, a 5xx, and above all a context-length
        # 400. That last one is a failure *this loop creates*: it accumulates
        # tool results a single-shot prompt never would, and the small windows
        # on self-hosted models are exactly the population the fallback exists
        # for. No prose has been written yet, so the blob prompt is still
        # available and is a better answer than an error card. Logged at error
        # because, unlike a tools rejection, this is not a settled property of
        # the provider — it wants looking at.
        log.error(
            "agent: turn failed upstream (model=%s) — falling back to the "
            "single-shot prompt: %s",
            cfg.model,
            exc,
        )
        raise AgenticUnavailable(f"upstream error: {exc}", code=CODE_UPSTREAM) from exc
    except httpx.RequestError as exc:
        # A connection reset, a read timeout, DNS. Same reasoning: nothing has
        # been written, and the other path may well reach the same server fine
        # with a smaller request.
        log.error(
            "agent: turn could not reach the provider (model=%s) — falling back "
            "to the single-shot prompt: %s",
            cfg.model,
            exc,
        )
        raise AgenticUnavailable(
            f"upstream request failed: {exc}", code=CODE_UNREACHABLE
        ) from exc
    finally:
        merged = merge_usage(run_usage.get("usage"), turn_usage.get("usage"))
        if merged is not None:
            run_usage["usage"] = merged


# ── Dispatch ────────────────────────────────────────────────────────────────


async def _dispatch(
    request: AgentRequest,
    caller: ToolCaller,
    call: PendingToolCall,
) -> str:
    """Run one tool call and return the content of its ``role: "tool"`` message.

    Never raises. Every failure — bad JSON, an unknown tool, a refusal, a
    timeout, a handler bug — comes back as prose the model reads and can act on,
    because an exception here would abort the turn and lose the run, while a
    sentence lets Koutsi try the next thing. That agreement with issue #42's
    error shaping is the whole reason the tools return
    *"No activity on 2026-07-14. Nearest rides: …"* instead of a 404.

    **Each call gets its own session**, opened by ``call_tool``, rather than
    sharing the run's. Sharing was the cheaper thing — one connection to one
    SQLite file instead of one per call — and it was wrong, because
    :data:`TOOL_TIMEOUT_S` cancels a call wherever it happens to be:

    * a cancellation landing mid-statement invalidates the connection, so every
      later use of the run's session raises ``PendingRollbackError`` — including
      the write of the answer and ``stream_into_db``'s own error handler; and
    * the ``rollback()`` that repairs *that* expires every ORM instance in the
      session, and an expired attribute needs IO to reload, which a plain
      attribute read cannot do under asyncio (``MissingGreenlet``). The run
      reads ``athlete`` on every later tool call and the blob fallback reads the
      analyzer's objects throughout — so the repair broke the degradation path
      the timeout exists to protect.

    A session nobody else holds has neither problem, and has them for no inputs
    rather than for the ones we thought to handle. The cost is a pooled
    connection checkout and one ``load_athlete`` per call — at most 24 per run
    against a local file whose engine is already cached, on a path that also
    opens a registry session per call for the consent check.
    """
    started = time.perf_counter()
    arguments, parse_error = call.parse_arguments()
    if parse_error is not None:
        _log_call(request, call, "bad_json", 0.0, arguments=None)
        return parse_error

    try:
        result: ToolResult = await asyncio.wait_for(
            call_tool(
                caller,
                call.name,
                arguments,
                # Deliberately **not** given a session or an athlete: see
                # `_dispatch`'s docstring. `call_tool` opens its own, which is
                # what keeps a cancelled call from reaching anything the run
                # still needs.
                #
                # The athlete's date, not the server's. The brief asserts one
                # ("Today is 2026-08-10 (Sunday) … in the athlete's own
                # timezone"), and a tool answering from a different one turns
                # "not due yet" into "missed" for anyone far enough from UTC.
                today=request.today,
            ),
            timeout=TOOL_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        _log_call(request, call, "timeout", (time.perf_counter() - started) * 1000, arguments)
        return (
            f"'{call.name}' took longer than {TOOL_TIMEOUT_S:.0f} seconds and was "
            "stopped. Try a narrower request — a shorter date range or a smaller "
            "limit — or answer with what you already have."
        )
    except Exception:  # pragma: no cover - call_tool is itself defensive
        log.exception("agent: dispatching %s failed unexpectedly", call.name)
        _log_call(request, call, "failed", (time.perf_counter() - started) * 1000, arguments)
        return (
            f"'{call.name}' failed unexpectedly. The failure has been logged; "
            "try a different tool or answer with what you already have."
        )

    _log_call(
        request,
        call,
        "ok" if result.ok else "error",
        result.duration_ms,
        arguments,
    )
    return _cap(result.text())


def _cap(content: str) -> str:
    """Truncate an oversized tool result, saying so explicitly.

    The marker matters more than the truncation: a model handed a silently
    shortened list assumes it saw everything and reports a confident wrong
    answer, which is worse than no answer because nothing downstream can tell it
    apart from a right one.
    """
    if len(content) <= MAX_TOOL_RESULT_CHARS:
        return content
    omitted = len(content) - MAX_TOOL_RESULT_CHARS
    return content[:MAX_TOOL_RESULT_CHARS] + TRUNCATION_MARKER.format(omitted=omitted)


def _log_call(
    request: AgentRequest,
    call: PendingToolCall,
    outcome: str,
    duration_ms: float,
    arguments: Optional[dict],
) -> None:
    """One structured line per tool call.

    ``call_tool`` already writes the security audit record (``openkoutsi.audit``,
    keyed on the principal). This is the operational twin of it: which surface
    asked, what it asked for, how long it took, how it went — the record you need
    to answer "why did this athlete's card take ninety seconds" without reading
    the audit log for a different purpose than it was built for.
    """
    log.info(
        "agent tool call feature=%s tool=%s outcome=%s duration_ms=%.1f arguments=%s",
        request.feature,
        call.name,
        outcome,
        duration_ms,
        json.dumps(arguments, default=str, ensure_ascii=False) if arguments is not None else "-",
    )
