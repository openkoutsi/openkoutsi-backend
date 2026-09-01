"""Instance capability gates (issue #56).

One place to ask "does this instance offer X", so a capability can be refused
everywhere it is reachable rather than only at the HTTP route that happens to
be the front door today.

That distinction is the whole point, and the codebase already has the lesson
written down twice. ``allow_personal_access_tokens`` is checked inside
``validate_personal_access_token`` rather than only in ``api/tokens``, because
a check on the issuance route would have left ``/mcp`` wide open; and
``allow_mcp_server`` is checked before the handshake rather than per tool, so a
disabled server never reports itself as present. Course recon gets the same
treatment: the switch refuses the *capability* — every course and bike
endpoint, the background matcher, the plan generator — not the entry point.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.registry_orm import InstanceSettings


async def course_recon_enabled(session: AsyncSession) -> bool:
    """Whether this instance offers course recon at all (issue #56).

    Defaults **off** when no settings row exists, unlike the MCP and token
    gates which default on. Those publish an interface over data the caller's
    credential already reaches; this one gates a feature whose distinguishing
    half needs a routing sidecar with tiles the self-hoster builds themselves.
    An instance that has never been configured has not consented to that, so
    absent reads as no.
    """
    instance = (await session.execute(select(InstanceSettings).limit(1))).scalar_one_or_none()
    return bool(instance and instance.allow_course_recon)
