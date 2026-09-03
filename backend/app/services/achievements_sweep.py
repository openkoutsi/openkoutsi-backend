"""Daily catch-up for athletes with a pending achievement recompute (issue #69).

The write paths mark ``athletes.achievements_dirty_at`` and return; something has
to settle it. ``GET /achievements`` does, on every read — but an athlete who
uploads from a head unit and doesn't open the app would otherwise have the inbox
message about their new badges wait until they did, and arrive dated whenever
that happened to be rather than near when the badge was actually earned.

So this sweep settles what the reads haven't. It is the only consumer that gates
on the flag: the reads cannot, because the achievements response needs progress
and streaks whatever the flag says, so they reconcile unconditionally anyway.
That is also why the flag is cheap to be approximate about — see the race note on
``achievements.recompute_achievements``.

The flag lives in each *per-user* DB while the user list lives in the *registry*,
so the sweep reads registry rows and then opens the affected users' sessions,
exactly as :mod:`backend.app.services.pat_expiry` already does. It runs as a
periodic task in ``lifespan`` beside that sweep and the bridge pollers — the
existing pattern for background work here, and why this needs no scheduler
dependency.

Idempotence comes from the flag, not the schedule: ``recompute_achievements``
clears it in the same commit as the rows it writes, and a reconcile over
unchanged data inserts nothing, so a sweep that runs twice — or that missed a day
— announces nothing twice. Running concurrently with a ``GET /achievements`` on
the same athlete contends for that user DB's single write lock; the engine sets a
30 s busy timeout, so that is a wait rather than a correctness problem, and
whichever pass loses has nothing left to write.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.models.registry_orm import User

log = logging.getLogger(__name__)

#: How often the sweeper wakes. Daily, matching `pat_expiry` — the badges it
#: settles are already visible to anyone who opens the app, so this only bounds
#: how long an *absent* athlete's inbox message waits.
SWEEP_INTERVAL_SECONDS = 24 * 60 * 60


async def _settle_user(user_id: str) -> int:
    """Reconcile every dirty athlete in one user's DB. Returns how many ran."""
    from backend.app.core.encryption import set_user_encryption_context
    from backend.app.db.user_session import get_user_session_factory
    from backend.app.models.user_orm import Athlete
    from backend.app.services.achievements import recompute_achievements

    # No per-user column the reconcile touches is encrypted today, but this is
    # what `deps.open_user_session` does before handing a session to a route, and
    # a sweep that skipped it would be the one caller that silently broke the day
    # one of them became so.
    set_user_encryption_context(user_id)

    settled = 0
    async with get_user_session_factory(user_id)() as session:
        athletes = (
            await session.execute(
                select(Athlete).where(Athlete.achievements_dirty_at.is_not(None))
            )
        ).scalars().all()
        for athlete in athletes:
            # Not the `_safe` wrapper: its rollback exists to protect a *caller*
            # that carries on using the session, and here the session is this
            # function's own. The per-user guard below is what isolates failures.
            await recompute_achievements(athlete.id, session, athlete=athlete)
            settled += 1
    return settled


def _has_database(user_id: str) -> bool:
    """Whether this user's database file exists yet.

    Self-serve signup writes the registry row and sends the confirmation email;
    ``_create_user_profile`` — and with it the database — only runs when the link
    is followed. Every pending or abandoned signup is therefore a live,
    undeleted user with no file, and there is no column that says so: an invited
    account never verifies an address either, so ``email_verified_at`` would read
    as "no database" for exactly the users who have one. The file is the fact.

    Deliberately the same ``settings.user_db_path`` that ``_get_user_engine``
    resolves, so this cannot decide one thing and the connection another.
    """
    return Path(settings.user_db_path(user_id)).exists()


async def run_achievements_sweep(registry_session: AsyncSession) -> int:
    """Settle every user's pending recomputes. Returns how many athletes ran.

    Deliberately opens every live user's DB rather than keeping a registry-side
    index of who is dirty: the flag belongs with the data it describes, and a
    once-a-day open of each SQLite file is the same cost the migration runner
    already pays on every boot. "Every live user" means every user who has a
    database — see ``_has_database`` for the ones who don't.
    """
    users = (
        await registry_session.execute(
            select(User.id).where(User.deleted_at.is_(None))
        )
    ).scalars().all()

    settled = 0
    for user_id in users:
        if not _has_database(user_id):
            # Nothing to settle: no database means no athlete and no activities.
            # Skipped rather than attempted because `_get_user_engine` creates no
            # file (issue #102), so opening a session for one of these raises
            # `unable to open database file` — which the guard below would log as
            # a full traceback, once per pending signup, every single day, until
            # the noise buried the genuinely broken database it exists to report.
            # Building the engine also costs one of its 256 cache slots, evicting
            # a real user's for an account that has nothing behind it.
            #
            # An account that activates between this check and the sweep's next
            # pass simply waits for that one — and any read of `GET /achievements`
            # settles it sooner anyway.
            log.debug("Achievement sweep skipped user %s: no database", user_id)
            continue
        try:
            settled += await _settle_user(user_id)
        except Exception:
            # One unreadable DB is not a reason to leave every remaining athlete
            # unsettled — and not clearing the flag means the next sweep retries.
            log.exception("Achievement sweep failed for user %s", user_id)
    return settled


async def achievements_sweep_once() -> None:
    """One sweep. The loop and the leader claim live in ``backend.main``."""
    from backend.app.db.registry import get_registry_session

    async for session in get_registry_session():
        settled = await run_achievements_sweep(session)
        if settled:
            log.info("Achievement sweep settled %d athlete(s)", settled)
        break
