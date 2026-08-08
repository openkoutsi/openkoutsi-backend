"""The MCP endpoint — JSON-RPC over HTTP at ``/mcp`` (issue #42).

What this is
------------
A minimal, stateless implementation of the MCP *Streamable HTTP* transport: one
``POST /mcp`` carrying one JSON-RPC 2.0 message. Four methods are served —
``initialize``, ``ping``, ``tools/list`` and ``tools/call`` — plus a 202 for any
notification, which is everything a tools-only server owes a client. Resources
and prompts are deliberately absent from the advertised capabilities rather than
stubbed, so a client never offers a user something that will fail.

No session id is issued and none is required: every request carries its own
credential and the server keeps nothing between them, which is the shape the spec
calls stateless and the shape a self-hosted single-process deployment wants.
Batching is refused — it was removed from the protocol in the 2025-06-18 revision.

Why the route-policy walk cannot help here, and what replaces it
----------------------------------------------------------------
Every other authenticated endpoint in this API is default-deny for a personal
access token, because :func:`backend.app.core.scopes.build_access_map` resolves
each route's declaration at app construction and ``get_current_user`` refuses
anything absent from the map. This endpoint cannot participate in that, for a
reason that is a property of the design rather than an oversight: **there is no
single scope a URL behind nine differently-scoped tools could honestly declare.**
``metrics:read`` on this path would lock out a token holding only ``plans:read``
that legitimately wants ``get_plan_status``; declaring nothing would close it to
tokens entirely. The scope a call needs is a property of the *tool*, and the tool
name is in the request body.

So this route resolves its credential itself, through
:func:`backend.app.core.auth.authenticate_bearer` — which still proves the token
is well-formed, live, unrevoked and enabled on this instance, and shares that
code with the ordinary resolver — and then defers every *authorization* decision
to :func:`backend.app.mcp.dispatch.call_tool`, where the tool's own declared
scopes are checked. The default-deny lives in
:mod:`backend.app.mcp.registry`: a tool that declares no scopes cannot be
registered, so it cannot exist to be called.

Two tests hold that in place. ``tests/unit/test_mcp_registry.py`` is the
registry's equivalent of the route walk. And ``test_pat_scopes.py`` asserts that
``authenticate_bearer`` has exactly **one** call site — so a second endpoint that
quietly steps outside the route policy is a test failure rather than a discovery.

Registered as an ordinary route rather than a mounted sub-application, which was
the first shape this took: a ``Mount`` matches only ``/mcp/…``, so bare ``/mcp``
answered with a 307 to the trailing-slash form, and an MCP client whose HTTP
layer does not follow redirects on POST would have seen an empty body. The API's
own convention is no trailing slash on a collection root; this follows it.

Exposure
--------
Shipping the endpoint is not the same as publishing it. It speaks only to a
credential this instance issued, so a self-hoster who does not want it reachable
from outside simply does not route ``/mcp`` through their reverse proxy — the
same decision they already make for the rest of the API. That narrows the
*interface*, not the exposure: the same data is reachable through the ordinary
REST routes, and what limits a credential is its scopes.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import authenticate_bearer
from backend.app.core.limiter import limiter
from backend.app.db.registry import get_registry_session
from backend.app.mcp.dispatch import ToolCaller, call_tool
from backend.app.mcp.registry import all_tools

log = logging.getLogger(__name__)

#: MCP revision this server implements. Echoed back from ``initialize``; a client
#: asking for a different one is told what it gets rather than refused, which is
#: what the spec asks for.
PROTOCOL_VERSION = "2025-06-18"

SERVER_INFO = {
    "name": "openkoutsi",
    "title": "openkoutsi coaching tools",
    "version": "1.0.0",
}

# JSON-RPC error codes. The application-level ones (a tool refusing, a tool
# failing) are deliberately *not* here: those come back as successful results
# with ``isError`` set, so the model can read them and carry on.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602


def _error(request_id: Any, code: int, message: str, status: int = 200) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        },
    )


def _result(request_id: Any, payload: dict) -> JSONResponse:
    return JSONResponse(
        content={"jsonrpc": "2.0", "id": request_id, "result": payload}
    )


async def _mcp_enabled(registry_session: AsyncSession) -> bool:
    """Whether this instance publishes the MCP server at all (issue #42).

    Checked before anything else, handshake included: a disabled endpoint that
    still completed `initialize` would let a client believe it had connected to
    a server that will refuse every useful call. Answered as a 404, which is
    what a not-published endpoint is, with a body saying so — the self-hoster
    who turned it off and forgot is the person most likely to be reading it.
    """
    from sqlalchemy import select

    from backend.app.models.registry_orm import InstanceSettings

    instance = (
        await registry_session.execute(select(InstanceSettings).limit(1))
    ).scalar_one_or_none()
    return instance is None or bool(instance.allow_mcp_server)


def _bearer(request: Request) -> Optional[str]:
    """The bearer value, or ``None``.

    The scheme is matched case-insensitively because RFC 7235 says auth-scheme
    is, and MCP clients differ on how they build the header. A client sending
    ``bearer okp_…`` would otherwise be told its credential was *missing* — an
    error message pointing at the token when the fault is one character of
    casing, which is a miserable thing to debug from the far end.
    """
    header = request.headers.get("Authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return value.strip() or None


_RPC_REQUEST_SCHEMA = {
    "type": "object",
    "required": ["jsonrpc", "method"],
    "properties": {
        "jsonrpc": {"const": "2.0", "description": "Always the string '2.0'."},
        "id": {
            "description": (
                "Correlation id, echoed back on the response. Omit for a "
                "notification, which is answered with 202 and no body."
            )
        },
        "method": {
            "type": "string",
            "enum": [
                "initialize",
                "notifications/initialized",
                "ping",
                "tools/list",
                "tools/call",
            ],
        },
        "params": {
            "type": "object",
            "description": (
                "Method parameters. For tools/call: {'name': <tool name>, "
                "'arguments': {…}}, where the accepted arguments come from that "
                "tool's inputSchema in tools/list."
            ),
        },
    },
    "additionalProperties": False,
}

_ENDPOINT_DESCRIPTION = """\
Model Context Protocol endpoint (issue #42) — the tool interface an AI coach
talks to, over JSON-RPC 2.0 rather than REST.

`initialize` and `ping` need no credential, and any JSON-RPC notification (a
message with no `id`) is answered with 202 and no body.
`tools/list` and `tools/call` are authenticated with the same
`Authorization: Bearer …` header as the rest of this API, accepting either a
session token or a personal access token.

**Personal access token scope:** decided per tool, not per route. Each tool
declares the read scopes it needs and `tools/list` reports them in its `_meta`;
a call whose credential lacks one comes back as a tool result explaining the
gap, not as a 403. Every published tool is read-only.
"""


def create_mcp_router() -> APIRouter:
    """Build the router. Included by ``backend.main.create_app`` at ``/mcp``."""
    router = APIRouter(tags=["mcp"])

    # Published in `openapi.json` so a self-hoster reading the API reference
    # discovers the endpoint exists, even though the interesting schemas — the
    # tools' own — live behind `tools/list` where an MCP client looks for them.
    # The body is described by hand: a JSON-RPC envelope's `params` differ per
    # method, so a pydantic model would have to be `Any` and would document less
    # than this does.
    @router.post(
        "/mcp",
        operation_id="mcpJsonRpc",
        summary="MCP JSON-RPC endpoint",
        description=_ENDPOINT_DESCRIPTION,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": _RPC_REQUEST_SCHEMA}},
            }
        },
    )
    # The one HTTP-level limit this endpoint needs, and it belongs here rather
    # than only in the dispatcher: `initialize`, `ping` and *failed*
    # authentication never reach a tool, and a failed authentication costs two
    # registry queries and an audit line each time. Brute force is not the
    # concern — `mint_token` is high-entropy and `verify_secret` is a sha256
    # compare — but audit-log flooding and registry load from an unauthenticated
    # caller are. Keyed by `principal_key`, so an authenticated caller is
    # counted by user and everyone else by address. Every other
    # credential-accepting router in this API declares a limit; this one no
    # longer is the exception.
    @limiter.limit("120/minute")
    async def rpc(
        request: Request,
        registry_session: AsyncSession = Depends(get_registry_session),
    ):
        if not await _mcp_enabled(registry_session):
            return _error(
                None,
                INVALID_REQUEST,
                "The MCP server is disabled on this instance. An administrator "
                "can enable it under Settings → allow_mcp_server.",
                status=404,
            )

        try:
            message = await request.json()
        except Exception:
            return _error(None, PARSE_ERROR, "Request body is not valid JSON.")

        if isinstance(message, list):
            return _error(
                None,
                INVALID_REQUEST,
                "JSON-RPC batching is not supported; send one message per request.",
            )
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error(None, INVALID_REQUEST, "Expected a JSON-RPC 2.0 message.")

        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return _error(request_id, INVALID_PARAMS, "'params' must be an object.")

        # ── Notifications ────────────────────────────────────────────────────
        #
        # A JSON-RPC message with no ``id`` is a notification: it expects no
        # response at all, and the transport spec is explicit that the server
        # answers 202 with **no body**. Handled here as a class rather than by
        # name, because the set is open — a client sends
        # ``notifications/cancelled`` when the user interrupts a tool call, and
        # matching only ``notifications/initialized`` sent every other one down
        # the authenticated path to be answered with a 401 error object. Replying
        # to a notification at all is a protocol violation; replying to it with
        # an error a stricter client may drop the connection over.
        #
        # Deliberately before the credential check. There is nothing to
        # authorize: a notification we do not act on reads nothing, changes
        # nothing and discloses nothing, and the spec requires the same 202
        # either way.
        if "id" not in message:
            return Response(status_code=202)

        # ── Methods reachable without a credential ───────────────────────────
        #
        # Handshake and liveness only. They disclose the tool *names* by way of
        # capabilities, which a client needs before it has anywhere to put a
        # token, and nothing about the athlete.
        if method == "initialize":
            return _result(
                request_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": SERVER_INFO,
                    "instructions": _INSTRUCTIONS,
                },
            )
        if method == "ping":
            return _result(request_id, {})

        token = _bearer(request)
        if token is None:
            return JSONResponse(
                status_code=401,
                content={
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": INVALID_REQUEST,
                        "message": (
                            "Missing credential. Send a personal access token as "
                            "'Authorization: Bearer okp_…'; create one in the web "
                            "app under Settings → Personal access tokens."
                        ),
                    },
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            ctx = await authenticate_bearer(
                token, registry_session, method="POST", path="/mcp"
            )
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": INVALID_REQUEST, "message": str(exc.detail)},
                },
                headers=getattr(exc, "headers", None) or {},
            )

        caller = ToolCaller.from_context(ctx)

        if method == "tools/list":
            # Every tool is listed, including ones this credential cannot call.
            # Hiding them would make a scope refusal look like a missing feature,
            # and the scopes each tool needs are in its `_meta` so a client can
            # explain the gap rather than discover it by failing.
            return _result(request_id, {"tools": [t.describe() for t in all_tools()]})

        if method == "tools/call":
            name = params.get("name")
            if not isinstance(name, str) or not name:
                return _error(request_id, INVALID_PARAMS, "'name' is required.")
            arguments = params.get("arguments", {})
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                return _error(request_id, INVALID_PARAMS, "'arguments' must be an object.")

            result = await call_tool(
                caller, name, arguments, registry_session=registry_session
            )
            content = {
                "content": [{"type": "text", "text": result.text()}],
                "isError": not result.ok,
            }
            if result.ok:
                content["structuredContent"] = result.data
            return _result(request_id, content)

        return _error(
            request_id,
            METHOD_NOT_FOUND,
            f"Method '{method}' is not supported. This server implements tools "
            f"only: initialize, ping, tools/list, tools/call.",
        )

    return router


_INSTRUCTIONS = """\
These tools read one athlete's cycling training data from their openkoutsi \
instance — the athlete who issued the credential in use, and no one else.

Start with get_training_status: it needs no arguments and tells you whether the \
interesting question is about load, freshness, plan adherence or something else. \
get_goal_progress is usually the second call, because the same fitness numbers \
mean opposite things before and after an event.

Everything is read-only. Nothing here can create, edit or delete anything, and \
raw per-second data streams are deliberately unavailable — the tools return \
computed aggregates instead. Where a figure is missing you will generally find a \
reason code beside it; treat those as findings rather than as gaps.

A tool may refuse because the credential's scopes do not cover it. That is fixed \
by issuing a new token in the web app, not by retrying.\
"""
