"""In-app notification service.

The single writer of user-facing messages. Today it only persists messages to
each recipient's per-user DB; `_dispatch_external` is the documented extension
point where future email / push / webhook delivery can be added without
touching any call sites.
"""
import json
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.user_session import get_user_session_factory, init_user_db
from backend.app.models.message_orm import Message
from backend.app.models.registry_orm import TeamMembership

log = logging.getLogger(__name__)

# ── Message types ────────────────────────────────────────────────────────────
TEAM_REQUEST = "team_request"   # new team signup awaiting superadmin approval
INVITE_USED = "invite_used"     # someone registered via an invite link
JOIN_REQUEST = "join_request"   # someone asked to join a team

# Reserved mailbox key for the superadmin, who has no registry user account.
SUPERADMIN_MAILBOX = "superadmin"


async def notify_user(user_id: str, type: str, data: dict) -> None:
    """Persist an in-app message to a single user's mailbox."""
    await init_user_db(user_id)
    async with get_user_session_factory(user_id)() as session:
        session.add(Message(type=type, data=data))
        await session.commit()
    await _dispatch_external(user_id, type, data)


async def notify_superadmin(type: str, data: dict) -> None:
    """Persist an in-app message to the superadmin mailbox."""
    await notify_user(SUPERADMIN_MAILBOX, type, data)


async def notify_team_admins(
    registry_session: AsyncSession, team_id: str, type: str, data: dict
) -> None:
    """Fan out an in-app message to every administrator of a team."""
    result = await registry_session.execute(
        select(TeamMembership).where(TeamMembership.team_id == team_id)
    )
    for membership in result.scalars().all():
        try:
            roles = json.loads(membership.roles)
        except (TypeError, ValueError):
            roles = []
        if "administrator" in roles:
            await notify_user(membership.user_id, type, data)


async def _dispatch_external(user_id: str, type: str, data: dict) -> None:
    """Extension point for external delivery (email, push, webhooks).

    Currently a no-op — messages are in-app only. Implement delivery here (e.g.
    look up the user's notification preferences and send an email) when external
    providers are added.
    """
    return None
