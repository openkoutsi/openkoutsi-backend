"""Daily clean-up of self-serve signups that were never confirmed.

Self-serve signup (issue #15) writes the ``users`` row *before* the address is
confirmed, so every abandoned attempt leaves a row behind: an email, a password
hash, ``email_verified_at`` NULL, and verification tokens that expired an hour
after they were sent. Nothing else ever removes it, so without this sweep those
rows accumulate forever and each one keeps personal data (the address) for an
account that never existed in any meaningful sense.

A pending signup is dropped once :data:`PENDING_SIGNUP_TTL` (30 days) has passed
since the **latest** signup attempt for it — re-signing up with the same address
issues a fresh token and so restarts the clock. A row still holding a live
verification token is never touched, whatever its age.

Only rows that self-serve signup can have produced are considered: an email, no
username, never verified. Invited and legacy accounts are keyed by username and
never verify an address, so the ``username IS NULL`` guard is what keeps this
from mistaking them for abandoned signups.

Runs as a periodic task in ``lifespan`` beside the other daily sweeps, under the
same background-work claim.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.models.registry_orm import EmailVerificationToken, User

log = logging.getLogger(__name__)

#: How long an unconfirmed signup is kept after its latest attempt.
PENDING_SIGNUP_TTL = timedelta(days=30)

#: How often the sweeper wakes. Daily — the TTL is measured in days.
SWEEP_INTERVAL_SECONDS = 24 * 60 * 60


def _aware(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


async def run_pending_signup_cleanup(
    registry_session: AsyncSession, now: Optional[datetime] = None
) -> int:
    """Delete every unconfirmed signup older than the TTL. Returns how many."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - PENDING_SIGNUP_TTL

    # Latest attempt per pending user, and whether any token is still live. One
    # grouped query rather than a lookup per candidate.
    latest_token = (
        select(
            EmailVerificationToken.user_id.label("user_id"),
            func.max(EmailVerificationToken.created_at).label("latest_created"),
            func.max(EmailVerificationToken.expires_at).label("latest_expires"),
        )
        .group_by(EmailVerificationToken.user_id)
        .subquery()
    )
    result = await registry_session.execute(
        select(User, latest_token.c.latest_created, latest_token.c.latest_expires)
        .outerjoin(latest_token, latest_token.c.user_id == User.id)
        .where(
            User.email.is_not(None),
            User.username.is_(None),
            User.email_verified_at.is_(None),
            User.created_at <= cutoff,
        )
    )

    stale: list[User] = []
    for user, latest_created, latest_expires in result.all():
        latest_created = _aware(latest_created)
        latest_expires = _aware(latest_expires)
        if latest_created is not None and latest_created > cutoff:
            continue  # Re-signed up recently: the clock restarted.
        if latest_expires is not None and latest_expires > now:
            continue  # A live link is out there; let it be followed.
        stale.append(user)

    if not stale:
        return 0

    for user in stale:
        # ORM delete, so the relationship cascades take the expired tokens with it.
        await registry_session.delete(user)
    await registry_session.commit()

    for user in stale:
        # A pending signup has no per-user DB — it is created on verification.
        # One can exist only if an activation failed half-way; remove it then,
        # and don't build an engine for the (normal) case where there is none.
        if Path(settings.user_data_dir(user.id)).exists():
            from backend.app.db.user_session import delete_user_db

            try:
                await delete_user_db(user.id)
            except Exception:
                log.exception("Failed to remove data dir of pending signup %s", user.id)

    return len(stale)


async def pending_signup_cleanup_once() -> None:
    """One sweep. The loop and the leader claim live in ``backend.main``."""
    from backend.app.db.registry import registry_session

    async with registry_session() as session:
        removed = await run_pending_signup_cleanup(session)
        if removed:
            log.info("Removed %d unconfirmed signup(s) older than %s", removed, PENDING_SIGNUP_TTL)
