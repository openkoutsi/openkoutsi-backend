"""Running one tool, with every check applied (issue #42).

:func:`call_tool` is the only way a tool ever runs. Both consumers — the internal
agent and the MCP transport — arrive here, which makes "the two doors reach the
same tools through the same checks" a fact about the code.

The order matters, and each step is here rather than in the tools:

1. **Resolve the tool.** Unknown names get a result naming the real ones, so a
   model that hallucinated one can recover in a turn.
2. **Check the rate limit** — external callers only (:mod:`backend.app.mcp.limits`).
   Before the refusals below, so calls that *cannot* succeed are counted.
3. **Check scopes.** Default-deny: the tool declares what it needs
   (:mod:`backend.app.mcp.registry`) and a credential holds all of it or is
   refused. A session credential carries ``scopes is None`` — full access.
4. **Check consent.** Reading health data back out through a tool is the same
   processing ``require_consent`` guards on ingestion.
5. **Validate arguments** against the tool's pydantic model, turning a schema
   violation into a sentence rather than a stack trace.
6. **Establish the per-user context** — ``open_user_session`` sets the encryption
   context and opens *that user's* database, then ``load_athlete`` finds their
   profile. Isolation here is physical, so this must never be reimplemented per
   tool, which is why :class:`ToolRun` hands the handler a session it did not
   open.
7. **Run, size-check and record.** Every invocation is audited with caller, tool,
   arguments, duration and outcome.

A handler raising :class:`~backend.app.mcp.errors.ToolError` produces an error
*result*; anything else escaping a handler is logged as a bug and returned as a
generic failure rather than propagated into the model's turn.

:class:`ToolCaller` deliberately carries no roles: with no ``is_admin`` to
consult, no tool can widen what it returns for an administrator. Administrative
data lives in the registry database, which no tool opens at all.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional, TYPE_CHECKING

from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core import audit
from backend.app.core.deps import load_athlete, open_user_session
from backend.app.core.encryption import set_user_encryption_context
from backend.app.mcp.errors import ToolAccessError, ToolError, ToolNotFound
from backend.app.mcp.limits import tool_limiter
from backend.app.mcp.registry import (
    Tool,
    get_published_tool,
    get_tool,
    published_tool_names,
    tool_names,
)
from backend.app.models.user_orm import Athlete

if TYPE_CHECKING:  # pragma: no cover - typing only, and a cycle if imported
    from backend.app.services.llm_client import ResolvedLlm

log = logging.getLogger(__name__)

#: Callers that are the server itself rather than a credential presented over
#: the network. Exempt from the rate limiter, nothing else.
INTERNAL = "agent"
SESSION = "session"
PAT = "pat"

#: A tool result larger than this is a bug in that tool's shaping, not a large
#: answer: the whole design forbids returning raw series, and nothing that is
#: genuinely an aggregate approaches 64 KiB. Enforced here so a future tool
#: cannot quietly start spending a model's entire context window.
MAX_RESULT_BYTES = 64 * 1024


@dataclass(frozen=True)
class ToolCaller:
    """Who is calling, reduced to what authorization actually needs.

    Deliberately not a :class:`~backend.app.core.auth.UserContext`: that carries
    ``roles``, and a tool must never be able to answer differently for an
    administrator. Build one with :meth:`from_context`.
    """

    user_id: str
    #: ``None`` for a session credential (full access); an explicit list for a
    #: personal access token.
    scopes: Optional[list[str]] = None
    kind: str = SESSION
    #: ``personal_access_tokens.id`` when this is a token — the principal the
    #: audit log keys on.
    token_id: Optional[str] = None

    @classmethod
    def from_context(cls, ctx) -> "ToolCaller":
        """Narrow a :class:`~backend.app.core.auth.UserContext` to a caller."""
        return cls(
            user_id=ctx.user_id,
            scopes=ctx.scopes,
            kind=PAT if ctx.is_pat else SESSION,
            token_id=ctx.token_id,
        )

    @classmethod
    def internal(cls, user_id: str) -> "ToolCaller":
        """The on-server agent acting for a user it has already authenticated."""
        return cls(user_id=user_id, scopes=None, kind=INTERNAL)

    @property
    def principal(self) -> str:
        """A stable label for logs: the token when there is one, else the user."""
        return f"token:{self.token_id}" if self.token_id else f"user:{self.user_id}"


@dataclass
class ToolRun:
    """Everything a handler is given. It opens nothing for itself."""

    caller: ToolCaller
    session: AsyncSession
    athlete: Athlete
    today: date = field(default_factory=date.today)

    #: The athlete's own resolved LLM, when the caller that dispatched this run
    #: had already resolved one and was entitled to (issue #72).
    #:
    #: The seam exists because ``resolve_llm_config`` and ``check_llm_access``
    #: both read the **registry** database, which no tool module may so much as
    #: name (``test_no_tool_module_reaches_the_registry_database``) — that test
    #: is what keeps an administrator's session from being served more than an
    #: ordinary athlete's, and it must not be weakened to fit a feature. So the
    #: config arrives already resolved, by the loop that was entitled to resolve
    #: it: ``llm_agent._run`` does it once per run via ``resolve_stream_setup``.
    #:
    #: ``None`` is the ordinary case and must always be handled — an external
    #: MCP caller has no entitled loop in front of it, so a tool that can use a
    #: model degrades to a deterministic answer rather than refusing.
    llm: Optional["ResolvedLlm"] = None

    #: The chat turn this call belongs to, when there is one (issue #72). A
    #: proposal has to be attached to the turn that offered it, or the athlete
    #: has nowhere to answer it. ``None`` everywhere else, and a proposal drafted
    #: without one is simply unreachable from the UI rather than an error.
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None


@dataclass(frozen=True)
class ToolContext:
    """What the caller knew and a tool cannot resolve for itself (issue #72).

    Carried from :func:`call_tool` to :class:`ToolRun` as one object so adding a
    second thing the caller must hand down does not mean threading a fourth
    positional argument through :func:`_run`.
    """

    llm: Optional["ResolvedLlm"] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None


@dataclass
class ToolResult:
    """The outcome of one invocation, in the shape both consumers want."""

    tool: str
    ok: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    duration_ms: float = 0.0

    def text(self) -> str:
        """The human/model-readable body: JSON on success, prose on failure."""
        if not self.ok:
            return self.error or "The tool failed."
        return json.dumps(self.data, default=str, ensure_ascii=False)


def _failure(tool: str, message: str, started: float) -> ToolResult:
    return ToolResult(
        tool=tool,
        ok=False,
        error=message,
        duration_ms=round((time.perf_counter() - started) * 1000, 1),
    )


def _format_validation_error(tool: Tool, exc: ValidationError) -> str:
    """Turn pydantic's report into something a model can act on."""
    problems = []
    for err in exc.errors():
        where = ".".join(str(p) for p in err["loc"]) or "(root)"
        problems.append(f"{where}: {err['msg']}")
    accepted = ", ".join(sorted(tool.arguments.model_fields)) or "(no arguments)"
    return (
        f"The arguments to '{tool.name}' were not valid — "
        + "; ".join(problems)
        + f". Accepted arguments: {accepted}."
    )


async def _consent_ok(user_id: str, registry_session: Optional[AsyncSession]) -> bool:
    from backend.app.api.consent import has_consent

    if registry_session is not None:
        return await has_consent(user_id, registry_session)

    # `registry_session()`, not the FastAPI dependency: driving a DI generator
    # by hand works only while it stays single-yield with its cleanup in a
    # `with`, and would silently stop meaning what it looks like if anyone added
    # a second yield or a post-yield commit.
    from backend.app.db.registry import registry_session as open_registry_session

    async with open_registry_session() as session:
        return await has_consent(user_id, session)


async def call_tool(
    caller: ToolCaller,
    name: str,
    arguments: Optional[dict[str, Any]] = None,
    *,
    session: Optional[AsyncSession] = None,
    athlete: Optional[Athlete] = None,
    registry_session: Optional[AsyncSession] = None,
    today: Optional[date] = None,
    llm: Optional["ResolvedLlm"] = None,
    conversation_id: Optional[str] = None,
    message_id: Optional[str] = None,
    published_only: bool = False,
) -> ToolResult:
    """Run one tool for one caller. Never raises for an ordinary failure.

    ``session``/``athlete`` may be supplied by a caller that already holds the
    user's session (the agent inside a request, the tests); otherwise this opens
    the user's own encrypted database. Handing over the session belonging to
    ``caller.user_id`` is the caller's job, which is why the default is to open
    it here from the caller's own id.

    ``registry_session`` is used only for the consent check.

    ``today`` is the calendar date the tools reckon from. *The athlete's* date is
    not the server's, and the six tools keying off it are the
    date-boundary-sensitive ones where being a day out turns "not due yet" into
    "missed". Callers that know the athlete's timezone should pass their local
    date; omitted, it falls back to the process's own.

    ``llm``, ``conversation_id`` and ``message_id`` are handed straight to
    :class:`ToolRun` — see its fields for why a tool cannot resolve any of them
    for itself.

    ``published_only`` narrows the resolvable set to
    :func:`~backend.app.mcp.registry.published_tools`, so the MCP transport
    refuses an internal tool by name instead of running something it does not
    list (issue #72).
    """
    started = time.perf_counter()
    arguments = arguments or {}

    tool = get_published_tool(name) if published_only else get_tool(name)
    if tool is None:
        error = ToolNotFound(
            name, published_tool_names() if published_only else tool_names()
        )
        audit.mcp_tool_call(
            tool=name,
            outcome=audit.UNKNOWN_TOOL,
            user_id=caller.user_id,
            token_id=caller.token_id,
            caller_kind=caller.kind,
            arguments=arguments,
            duration_ms=0.0,
        )
        return _failure(name, error.rendered(), started)

    def record(outcome: str, duration_ms: float) -> None:
        audit.mcp_tool_call(
            tool=tool.name,
            outcome=outcome,
            user_id=caller.user_id,
            token_id=caller.token_id,
            caller_kind=caller.kind,
            arguments=arguments,
            duration_ms=duration_ms,
        )

    def fail(outcome: str, message: str) -> ToolResult:
        result = _failure(tool.name, message, started)
        record(outcome, result.duration_ms)
        return result

    # ── 2. Rate limit ────────────────────────────────────────────────────────
    #
    # Ahead of the refusal paths, not behind them. Counting only the calls that
    # were going to succeed inverts the intent: the runaway loop most likely to
    # actually happen is a client retrying against a credential that *cannot*
    # succeed, and every one of those iterations writes an audit record and (for
    # the consent check) costs a registry round-trip.
    if caller.kind != INTERNAL:
        allowed, retry_after = tool_limiter.check(caller.user_id)
        if not allowed:
            return fail(
                audit.RATE_LIMITED,
                f"Too many tool calls in the last minute. Retry in about "
                f"{retry_after} s, or make fewer, broader calls.",
            )

    # ── 3. Scopes ────────────────────────────────────────────────────────────
    missing = tool.missing_scopes(caller.scopes)
    if missing:
        return fail(audit.DENIED_SCOPE, ToolAccessError(tool.name, missing).rendered())

    # ── 4. Consent ───────────────────────────────────────────────────────────
    if not await _consent_ok(caller.user_id, registry_session):
        return fail(
            audit.DENIED_CONSENT,
            "This account has not accepted the current data-processing policy, "
            "so its training data cannot be read. The account holder needs to "
            "accept it in the web app before any tool here can answer.",
        )

    # ── 5. Arguments ─────────────────────────────────────────────────────────
    try:
        parsed = tool.arguments.model_validate(arguments)
    except ValidationError as exc:
        return fail(audit.BAD_ARGUMENTS, _format_validation_error(tool, exc))

    # ── 6–7. Context, run, record ────────────────────────────────────────────
    context = ToolContext(
        llm=llm, conversation_id=conversation_id, message_id=message_id
    )
    try:
        if session is not None:
            # A caller-supplied session is the caller's responsibility, but it
            # must not be a *silent* one: `caller.user_id` is what the scope
            # check, the consent check and every audit record key on, and
            # nothing ties it to the session handed in. Deriving the key from
            # `caller.user_id` here turns a mismatch into a decryption failure
            # instead of one athlete's data returned under another's name. One
            # HKDF derivation; the `else` branch gets the same thing from
            # `open_user_session`.
            set_user_encryption_context(caller.user_id)
            result = await _run(
                tool, caller, session, athlete, parsed, today, context
            )
        else:
            async with open_user_session(caller.user_id) as owned:
                result = await _run(
                    tool, caller, owned, None, parsed, today, context
                )
    except ToolError as exc:
        return fail(audit.TOOL_ERROR, exc.rendered())
    except Exception:  # pragma: no cover - defensive; a handler bug, not a miss
        log.exception("mcp tool %s failed for %s", tool.name, caller.principal)
        return fail(
            audit.FAILED,
            f"'{tool.name}' failed unexpectedly. The failure has been logged; "
            "try a different tool or a narrower question.",
        )

    payload = result.model_dump(mode="json")
    encoded = json.dumps(payload, default=str, ensure_ascii=False)
    if len(encoded.encode()) > MAX_RESULT_BYTES:
        log.error(
            "mcp tool %s produced %d bytes, over the %d byte bound",
            tool.name,
            len(encoded.encode()),
            MAX_RESULT_BYTES,
        )
        return fail(
            audit.OVERSIZED,
            f"'{tool.name}' produced a response too large to return. Narrow the "
            "request — a shorter window, a smaller 'limit' — and try again.",
        )

    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    record(audit.OK, duration_ms)
    return ToolResult(tool=tool.name, ok=True, data=payload, duration_ms=duration_ms)


async def _run(
    tool: Tool,
    caller: ToolCaller,
    session: AsyncSession,
    athlete: Optional[Athlete],
    parsed: BaseModel,
    today: Optional[date] = None,
    context: Optional[ToolContext] = None,
) -> BaseModel:
    """Resolve the athlete and hand the handler its :class:`ToolRun`."""
    from fastapi import HTTPException

    if athlete is None:
        try:
            athlete = await load_athlete(caller.user_id, session)
        except HTTPException as exc:
            # `load_athlete` speaks HTTP because every other caller is a route.
            # Here it means the account exists but was never onboarded, which is
            # a fact about the data and belongs in the answer.
            raise ToolError(
                "This account has no athlete profile yet, so there is no "
                "training data to read. It is created by completing the setup "
                "wizard in the web app."
            ) from exc

    context = context or ToolContext()
    run = ToolRun(
        caller=caller,
        session=session,
        athlete=athlete,
        llm=context.llm,
        conversation_id=context.conversation_id,
        message_id=context.message_id,
    )
    if today is not None:
        run.today = today
    return await tool.handler(run, parsed)
