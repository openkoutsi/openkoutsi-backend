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
from backend.app.models.registry_orm import User

log = logging.getLogger(__name__)

# ── Message types ────────────────────────────────────────────────────────────
INVITE_USED = "invite_used"     # someone registered via an invite link
INBOUND_EMAIL = "inbound_email"  # mail to the operator address (issue #38)


async def notify_user(user_id: str, type: str, data: dict) -> None:
    """Persist an in-app message to a single user's mailbox."""
    await init_user_db(user_id)
    async with get_user_session_factory(user_id)() as session:
        session.add(Message(type=type, data=data))
        await session.commit()
    await _dispatch_external(user_id, type, data)


async def notify_admins(
    registry_session: AsyncSession, type: str, data: dict
) -> int:
    """Fan out an in-app message to every instance administrator.

    Returns the number of administrators the message was delivered to.
    """
    result = await registry_session.execute(
        select(User).where(User.deleted_at.is_(None))
    )
    delivered = 0
    for user in result.scalars().all():
        try:
            roles = json.loads(user.roles) if user.roles else []
        except (TypeError, ValueError):
            roles = []
        if "administrator" in roles:
            await notify_user(user.id, type, data)
            delivered += 1
    return delivered


async def _dispatch_external(user_id: str, type: str, data: dict) -> None:
    """Extension point for external delivery (email, push, webhooks).

    Currently a no-op — messages are in-app only. Implement delivery here (e.g.
    look up the user's notification preferences and send an email) when external
    providers are added.
    """
    return None
