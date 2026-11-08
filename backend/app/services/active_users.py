"""Daily-active-user recording.

A user counts as active on a UTC day once they load the dashboard that day. The
web app calls ``POST /api/metrics/catch-up`` on every dashboard load, so that is
where :func:`record_active_day` is called from.
"""
import logging
from datetime import date, datetime, timezone

from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.registry_orm import UserActiveDay

log = logging.getLogger(__name__)


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


async def record_active_day(
    user_id: str, session: AsyncSession, day: date | None = None
) -> None:
    """Mark *user_id* active on *day* (default: today, UTC). Idempotent.

    Best-effort: a failure is logged and swallowed, because a missed count must
    never cost the user their dashboard.
    """
    try:
        await session.execute(
            insert(UserActiveDay)
            .values(user_id=user_id, day=day or utc_today())
            .on_conflict_do_nothing()
        )
        await session.commit()
    except Exception:
        await session.rollback()
        log.warning("Could not record active day for user %s", user_id, exc_info=True)
