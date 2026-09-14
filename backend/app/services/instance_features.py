"""Instance capability gates (issue #56).

One place to ask "does this instance offer X", so a capability can be refused
everywhere it is reachable rather than only at the HTTP route that happens to
be the front door today.

The codebase has the lesson twice already: ``allow_personal_access_tokens`` is
checked inside ``validate_personal_access_token`` rather than only in
``api/tokens``, because a check on the issuance route would have left ``/mcp``
open; and ``allow_mcp_server`` is checked before the handshake rather than per
tool. Course recon gets the same treatment — the switch refuses the *capability*
(every course and bike endpoint, the background matcher, the plan generator), not
the entry point.
"""

from __future__ import annotations

from typing import Optional

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


async def signup_halt(session: AsyncSession) -> tuple[bool, Optional[str]]:
    """Whether self-serve signup is paused right now, and the admin's reason.

    The exception to this module's rule. Every gate above refuses a *capability*
    wherever it is reachable; this one refuses a single door and says so out
    loud. ``allow_self_signup`` already decides whether the instance offers
    self-serve signup at all — a standing policy — and what this adds is a
    temporary stop for when the constraint is capacity instead: a resource
    bottleneck, a provider's API application limits. Invitations keep redeeming
    and verification links already emailed still activate, because what a halt
    protects is the rate at which strangers arrive, and an invitation is the
    admin's own deliberate act.

    Returns the flag and the reason together so a caller that needs both — and
    the only one that refuses is the only one that needs them — pays for one
    query rather than two.
    """
    instance = (await session.execute(select(InstanceSettings).limit(1))).scalar_one_or_none()
    if instance is None or not instance.signups_halted:
        return False, None
    return True, instance.signup_halt_reason
