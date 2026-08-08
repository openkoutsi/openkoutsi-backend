"""Running one tool, with every check applied (issue #42).

:func:`call_tool` is the only way a tool ever runs. Both consumers — the internal
agent and the MCP transport — arrive here, which is what makes "the two doors
reach the same tools through the same checks" a fact about the code rather than
a hope.

The order matters, and each step is here rather than in the tools for a reason:

1. **Resolve the tool.** Unknown names get a result naming the real ones, so a
   model that hallucinated a tool can recover in one turn instead of retrying.
2. **Check scopes.** Default-deny: the tool declares what it needs
   (:mod:`backend.app.mcp.registry`) and a credential either holds all of it or
   is refused. A session credential carries ``scopes is None`` — full access,
   the same meaning it has everywhere else in this codebase.
3. **Check consent.** ``require_consent`` guards the *ingestion* paths; reading
   health data back out through a tool is the same processing, so the same gate
   applies per invocation.
4. **Check the rate limit** — for external callers only; see
   :mod:`backend.app.mcp.limits`.
5. **Validate arguments** against the tool's pydantic model, turning a schema
   violation into a sentence rather than a stack trace.
6. **Establish the per-user context** — ``open_user_session`` sets the encryption
   context and opens *that user's* database, then ``load_athlete`` finds their
   profile in it. Isolation here is physical, so this is the step that must never
   be reimplemented per tool, and it is why :class:`ToolRun` hands the handler a
   session it did not open.
7. **Run, size-check and record.** Every invocation is audited with caller, tool,
   arguments, duration and outcome.

A handler that raises :class:`~backend.app.mcp.errors.ToolError` produces an
error *result*; anything else escaping a handler is a bug, and is logged as one
and returned as a generic failure rather than propagated into the model's turn.

Note what :class:`ToolCaller` does **not** carry: roles. There is no ``is_admin``
to consult, so no tool can widen what it returns for an administrator, and the
admin-data exclusion the issue asks for is structural rather than a rule someone
has to remember. Administrative data lives in the registry database, which no
tool opens at all.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core import audit
from backend.app.core.deps import load_athlete, open_user_session
from backend.app.mcp.errors import ToolAccessError, ToolError, ToolNotFound
from backend.app.mcp.limits import tool_limiter
from backend.app.mcp.registry import Tool, get_tool, tool_names
from backend.app.models.user_orm import Athlete

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

    from backend.app.db.registry import get_registry_session

    agen = get_registry_session()
    session = await agen.__anext__()
    try:
        return await has_consent(user_id, session)
    finally:
        await agen.aclose()


async def call_tool(
    caller: ToolCaller,
    name: str,
    arguments: Optional[dict[str, Any]] = None,
    *,
    session: Optional[AsyncSession] = None,
    athlete: Optional[Athlete] = None,
    registry_session: Optional[AsyncSession] = None,
) -> ToolResult:
    """Run one tool for one caller. Never raises for an ordinary failure.

    ``session``/``athlete`` may be supplied by a caller that already holds the
    user's session (the agent inside a request, the tests); when they are not,
    this opens the user's own encrypted database and finds their athlete in it.
    Passing someone else's session is not a hole this can close — it is the
    caller's job to hand over the session belonging to ``caller.user_id``, which
    is precisely why the default is to open it here from the caller's own id.

    ``registry_session`` is used only for the consent check; omitted, one is
    opened and closed around that check.
    """
    started = time.perf_counter()
    arguments = arguments or {}

    tool = get_tool(name)
    if tool is None:
        error = ToolNotFound(name, tool_names())
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

    # ── 2. Scopes ────────────────────────────────────────────────────────────
    missing = tool.missing_scopes(caller.scopes)
    if missing:
        return fail(audit.DENIED_SCOPE, ToolAccessError(tool.name, missing).rendered())

    # ── 3. Consent ───────────────────────────────────────────────────────────
    if not await _consent_ok(caller.user_id, registry_session):
        return fail(
            audit.DENIED_CONSENT,
            "This account has not accepted the current data-processing policy, "
            "so its training data cannot be read. The account holder needs to "
            "accept it in the web app before any tool here can answer.",
        )

    # ── 4. Rate limit ────────────────────────────────────────────────────────
    if caller.kind != INTERNAL:
        allowed, retry_after = tool_limiter.check(caller.user_id)
        if not allowed:
            return fail(
                audit.RATE_LIMITED,
                f"Too many tool calls in the last minute. Retry in about "
                f"{retry_after} s, or make fewer, broader calls.",
            )

    # ── 5. Arguments ─────────────────────────────────────────────────────────
    try:
        parsed = tool.arguments.model_validate(arguments)
    except ValidationError as exc:
        return fail(audit.BAD_ARGUMENTS, _format_validation_error(tool, exc))

    # ── 6–7. Context, run, record ────────────────────────────────────────────
    try:
        if session is not None:
            result = await _run(tool, caller, session, athlete, parsed)
        else:
            async with open_user_session(caller.user_id) as owned:
                result = await _run(tool, caller, owned, None, parsed)
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

    return await tool.handler(ToolRun(caller=caller, session=session, athlete=athlete), parsed)
