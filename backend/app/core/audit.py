"""Structured audit logging for scoped-credential activity (issues #46, #42).

Every request a PAT authenticates, and every MCP tool invocation, is recorded
here: principal, what was reached, and the outcome. Deliberately **structured
logs, not the shared usage DB** — invocation records against one person's health
data do not belong in a database shared across users, and a self-hoster's log
pipeline is already where the rest of this instance's operational record lives.

Records go to the ``openkoutsi.audit`` logger with the fields attached as
``extra``, so a JSON formatter emits them as fields while the default formatter
still prints a readable line.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger("openkoutsi.audit")

# Outcomes. `denied_scope` and `revoked` are separate on purpose: a caller
# presenting a credential we withdrew is a different event from one presenting a
# credential that never existed, and the whole reason dead token rows are
# retained is to keep those two distinguishable.
OK = "ok"
DENIED_SCOPE = "denied_scope"
DENIED_ROUTE = "denied_route"
UNKNOWN_TOKEN = "unknown_token"
BAD_SECRET = "bad_secret"
REVOKED = "revoked"
EXPIRED = "expired"
DISABLED = "disabled"

# Tool-layer outcomes (issue #42). `denied_scope` above is shared: a refusal is
# a refusal whether it happened at a route or at a tool.
DENIED_CONSENT = "denied_consent"
RATE_LIMITED = "rate_limited"
BAD_ARGUMENTS = "bad_arguments"
UNKNOWN_TOOL = "unknown_tool"
TOOL_ERROR = "tool_error"
OVERSIZED = "oversized"
FAILED = "failed"


def pat_request(
    *,
    outcome: str,
    token_id: Optional[str],
    user_id: Optional[str] = None,
    method: Optional[str] = None,
    path: Optional[str] = None,
    scope: Optional[str] = None,
) -> None:
    """Record one personal-access-token request."""
    log.info(
        "pat %s token=%s user=%s %s %s",
        outcome,
        token_id or "-",
        user_id or "-",
        method or "-",
        path or "-",
        extra={
            "event": "pat_request",
            "pat_outcome": outcome,
            "pat_token_id": token_id,
            "pat_user_id": user_id,
            "http_method": method,
            "http_path": path,
            "required_scope": scope,
        },
    )


def mcp_tool_call(
    *,
    tool: str,
    outcome: str,
    user_id: str,
    token_id: Optional[str] = None,
    caller_kind: str = "session",
    arguments: Optional[dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> None:
    """Record one MCP tool invocation (issue #42).

    Arguments are recorded because without them the record cannot answer the
    question it exists for — "what did this credential read?" — and because a
    tool's arguments are dates, ids and window lengths rather than content.
    The *results* are never logged: those are the health data itself.

    ``caller_kind`` separates the in-process agent from a credential presented
    over the network, so the two can be counted apart.
    """
    log.info(
        "mcp %s tool=%s caller=%s user=%s token=%s %.1fms",
        outcome,
        tool,
        caller_kind,
        user_id,
        token_id or "-",
        duration_ms,
        extra={
            "event": "mcp_tool_call",
            "mcp_outcome": outcome,
            "mcp_tool": tool,
            "mcp_caller_kind": caller_kind,
            "mcp_arguments": arguments or {},
            "mcp_duration_ms": duration_ms,
            "pat_token_id": token_id,
            "pat_user_id": user_id,
        },
    )


def pat_admin_revoke(
    *, token_id: str, user_id: str, admin_user_id: str
) -> None:
    """Record an administrator revoking someone else's token.

    On a self-hosted instance the admin holds ``ENCRYPTION_KEY`` and root on the
    box; they could already open ``registry.db`` and delete the row. The endpoint
    does not widen what an admin can do — it moves the action out of a shell and
    into this log.
    """
    log.warning(
        "pat admin_revoke token=%s user=%s by=%s",
        token_id,
        user_id,
        admin_user_id,
        extra={
            "event": "pat_admin_revoke",
            "pat_token_id": token_id,
            "pat_user_id": user_id,
            "admin_user_id": admin_user_id,
        },
    )
