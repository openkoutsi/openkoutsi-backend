"""Daily expiry sweep for personal access tokens (issue #46).

A one-year ceiling without warnings just relocates the outage. A token that dies
silently takes a nightly backup or a head-unit sync with it, and the first the
user hears of it is a broken integration weeks later. So this sweep warns at
seven days, again at one day, and once more when the token has actually run out.

**Each stage fires exactly once.** ``personal_access_tokens.last_expiry_notice``
records the stage already sent, so a sweep running daily — or twice after a
restart — does not re-notify. Without that column this would be a daily nag,
which is worse than silence.

Tokens live in the *registry* DB and the inbox in each *per-user* DB, so the
sweep reads registry rows and then opens the affected users' sessions, exactly as
:mod:`backend.app.services.notifications` already does. It runs as a periodic
task in ``lifespan`` beside the Strava and Wahoo bridge pollers — the existing
pattern for background work here, and why this needs no scheduler dependency. It
inherits their single-process assumption: two app processes would double-notify,
and ``last_expiry_notice`` is the mitigation.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.models.registry_orm import PersonalAccessToken, User
from backend.app.services import notifications
from backend.app.services import personal_access_tokens as pat

log = logging.getLogger(__name__)

EXPIRING_7D = "expiring_7d"
EXPIRING_1D = "expiring_1d"
EXPIRED = "expired"

#: Stages in the order they occur. A stage is only sent when it is *later* than
#: whatever was last sent, so a sweep that missed a day skips straight to the
#: current stage rather than replaying the ones it slept through.
STAGE_ORDER: tuple[str, ...] = (EXPIRING_7D, EXPIRING_1D, EXPIRED)

#: How often the sweeper wakes. Daily, matching the stage granularity.
SWEEP_INTERVAL_SECONDS = 24 * 60 * 60

#: Key in the athlete's ``app_settings`` opting *out* of the expiry email. The
#: inbox message is unconditional; only email is opt-out.
EMAIL_OPT_OUT_SETTING = "pat_expiry_emails"


def _aware(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def stage_for(
    token: PersonalAccessToken, now: Optional[datetime] = None
) -> Optional[str]:
    """The stage ``token`` is currently at, or ``None`` if it is not due one.

    A revoked token gets nothing: the user (or an admin) already ended it
    deliberately, and telling them it has now also expired is noise.
    """
    now = now or datetime.now(timezone.utc)
    if token.revoked_at is not None:
        return None
    expires_at = _aware(token.expires_at)
    if expires_at is None:
        return None
    remaining = expires_at - now
    if remaining <= timedelta(0):
        return EXPIRED
    if remaining <= timedelta(days=1):
        return EXPIRING_1D
    if remaining <= timedelta(days=7):
        return EXPIRING_7D
    return None


def _is_new_stage(stage: str, already_sent: Optional[str]) -> bool:
    if already_sent is None:
        return True
    if already_sent not in STAGE_ORDER:
        return True
    return STAGE_ORDER.index(stage) > STAGE_ORDER.index(already_sent)


def _days_left(token: PersonalAccessToken, now: datetime) -> Optional[int]:
    expires_at = _aware(token.expires_at)
    if expires_at is None or expires_at <= now:
        return None
    return max(1, round((expires_at - now).total_seconds() / 86400))


async def _email_opted_in(user_id: str) -> bool:
    """Whether this user still wants expiry email (default: yes)."""
    from backend.app.db.user_session import get_user_session_factory
    from backend.app.models.user_orm import Athlete

    try:
        async with get_user_session_factory(user_id)() as session:
            result = await session.execute(
                select(Athlete).where(Athlete.global_user_id == user_id)
            )
            athlete = result.scalar_one_or_none()
    except Exception:
        log.exception("Could not read expiry-email preference for user %s", user_id)
        return False
    if athlete is None:
        return False
    app_settings = athlete.app_settings or {}
    return app_settings.get(EMAIL_OPT_OUT_SETTING, True) is not False


async def _send_email(user: User, token: PersonalAccessToken, days_left: Optional[int]) -> None:
    """Best-effort expiry email. Silent when unconfigured or opted out.

    Guarded against ``Exception``, not just ``EmailError``, and with provider
    construction *inside* the guard: ``get_email_provider()`` raises a bare
    ``ValueError`` for an unrecognised ``EMAIL_PROVIDER``, so a single typo in
    the instance config would otherwise take the whole sweep down. This is the
    optional channel on top of an inbox message that has already been written —
    nothing it can do is worth failing the sweep for.
    """
    if not user.email or user.email_verified_at is None:
        return
    try:
        from backend.app.services.email import get_email_provider, send_token_expiry_email

        provider = get_email_provider()
        if not provider.is_configured:
            return
        if not await _email_opted_in(user.id):
            return
        await send_token_expiry_email(
            provider,
            to=user.email,
            token_name=token.name,
            days_left=days_left,
            manage_url=f"{settings.frontend_url}/settings",
        )
    except Exception:
        log.exception("Failed to send token-expiry email for token %s", token.id)


async def run_expiry_sweep(
    registry_session: AsyncSession, now: Optional[datetime] = None
) -> int:
    """Notify every token that has reached a new stage. Returns how many were sent."""
    now = now or datetime.now(timezone.utc)

    horizon = now + timedelta(days=7)
    # `EXPIRED` is terminal, so a token already marked with it can never reach a
    # new stage. Excluding those in SQL rather than in `_is_new_stage` keeps this
    # query bounded by *live* tokens instead of by the instance's entire history
    # of naturally-expired ones — dead rows are deliberately retained, so without
    # this the daily sweep would load and discard a set that only ever grows.
    result = await registry_session.execute(
        select(PersonalAccessToken).where(
            PersonalAccessToken.revoked_at.is_(None),
            PersonalAccessToken.expires_at <= horizon,
            PersonalAccessToken.last_expiry_notice.is_distinct_from(EXPIRED),
        )
    )
    tokens = list(result.scalars().all())
    if not tokens:
        return 0

    user_ids = {token.user_id for token in tokens}
    users_result = await registry_session.execute(
        select(User).where(User.id.in_(user_ids), User.deleted_at.is_(None))
    )
    users_by_id = {user.id: user for user in users_result.scalars()}

    sent = 0
    for token in tokens:
        stage = stage_for(token, now)
        if stage is None or not _is_new_stage(stage, token.last_expiry_notice):
            continue
        user = users_by_id.get(token.user_id)
        if user is None:
            continue

        days_left = _days_left(token, now)
        payload = {
            "token_id": token.id,
            "name": token.name,
            "stage": stage,
            "days_left": days_left,
            "expires_at": _aware(token.expires_at).isoformat(),
        }
        try:
            await notifications.notify_user(
                user.id,
                notifications.PAT_EXPIRED if stage == EXPIRED else notifications.PAT_EXPIRING,
                payload,
            )
        except Exception:
            # A mailbox that could not be written is not a reason to leave every
            # remaining token unwarned — and not marking the stage means the next
            # sweep retries this one.
            log.exception("Failed to notify user %s about token %s", user.id, token.id)
            continue

        # Mark and commit *immediately* after the inbox write, and before the
        # email. `notify_user` commits to the user's own DB, so the message is
        # already durable; anything that threw between here and a commit at the
        # end of the loop would roll back this mark while leaving that message
        # in place, and the user would be re-notified every day thereafter —
        # precisely the nag `last_expiry_notice` exists to prevent, arriving
        # exactly when the mail path is broken and the warning matters most.
        token.last_expiry_notice = stage
        await registry_session.commit()
        sent += 1

        await _send_email(user, token, days_left)

    return sent


async def pat_expiry_sweep_once() -> None:
    """One sweep. The loop and the leader claim live in ``backend.main``.

    Note the ordering change this brought with it: the sweep used to run
    immediately on boot and then sleep, so a restart loop could re-send. It now
    runs on the same schedule as everything else under the claim, and
    `last_expiry_notice` remains the guard against a duplicate notice either
    way.
    """
    from backend.app.db.registry import get_registry_session

    async for session in get_registry_session():
        sent = await run_expiry_sweep(session)
        if sent:
            log.info("Token expiry sweep sent %d notification(s)", sent)
        break
