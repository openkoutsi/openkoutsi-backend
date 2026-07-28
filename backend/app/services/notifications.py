"""In-app notification service.

The single writer of user-facing messages. Today it only persists messages to
each recipient's per-user DB; `_dispatch_external` is the documented extension
point where future email / push / webhook delivery can be added without
touching any call sites.

Message text is rendered here, once, from the type and its payload — see
`backend.app.services.message_text`. Being the single writer is what makes that
safe: no caller has to remember to supply copy, and no message can reach a
mailbox without it.
"""
import json
import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.user_session import get_user_session_factory, init_user_db
from backend.app.models.message_orm import Message
from backend.app.models.registry_orm import User
from backend.app.services.message_text import DEFAULT_LOCALE, render

log = logging.getLogger(__name__)

# ── Message types ────────────────────────────────────────────────────────────
INVITE_USED = "invite_used"                     # someone registered via an invite link
ACHIEVEMENT_UNLOCKED = "achievement_unlocked"   # one or more achievement tiers earned


async def notify_user(
    user_id: str,
    type: str,
    data: dict,
    title: Optional[str] = None,
    body: Optional[str] = None,
) -> None:
    """Persist an in-app message to a single user's mailbox.

    Copy comes from `message_text.render` unless the caller passes its own,
    which is the escape hatch for one-off messages that aren't type-driven
    (an admin broadcast, say).
    """
    if title is None or body is None:
        rendered = render(type, data)
        title = title if title is not None else rendered.title
        body = body if body is not None else rendered.body

    await init_user_db(user_id)
    async with get_user_session_factory(user_id)() as session:
        session.add(
            Message(type=type, data=data, title=title, body=body, locale=DEFAULT_LOCALE)
        )
        await session.commit()
    await _dispatch_external(user_id, type, data)


async def notify_admins(
    registry_session: AsyncSession,
    type: str,
    data: dict,
    title: Optional[str] = None,
    body: Optional[str] = None,
) -> None:
    """Fan out an in-app message to every instance administrator."""
    result = await registry_session.execute(
        select(User).where(User.deleted_at.is_(None))
    )
    for user in result.scalars().all():
        try:
            roles = json.loads(user.roles) if user.roles else []
        except (TypeError, ValueError):
            roles = []
        if "administrator" in roles:
            await notify_user(user.id, type, data, title=title, body=body)


async def _dispatch_external(user_id: str, type: str, data: dict) -> None:
    """Extension point for external delivery (email, push, webhooks).

    Currently a no-op — messages are in-app only. Implement delivery here (e.g.
    look up the user's notification preferences and send an email) when external
    providers are added.
    """
    return None
