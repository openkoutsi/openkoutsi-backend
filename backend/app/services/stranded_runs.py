"""Settling LLM runs that will never finish (issue #91).

Three surfaces write a ``pending`` status and let a background task settle it:
the training status (``athletes.training_status_status``), goal guidance
(``goals.guidance_status``) and the activity analysis
(``activities.analysis_status``). "Still running" is not something any of them
can observe — the row is the only record — so it is inferred from a clock, and
this module owns both halves of that inference:

* :func:`pending_timed_out` — the shared age check the routers apply on read.
  One constant for all three, where there used to be two copies of it and, for
  the activity analysis, no check at all.
* :func:`settle_stranded_runs` — the startup sweep. A ``pending`` row at boot is
  by definition from a process that is gone: the auto-analyse paths run under
  ``asyncio.create_task`` (cancelled at loop shutdown) and ``trigger_analysis``
  under ``BackgroundTasks`` (waited on only up to uvicorn's graceful-shutdown
  timeout), so nothing survives a restart. An ordinary redeploy is therefore
  enough to strand whatever was in flight, and
  :func:`~backend.app.services.llm_streaming.failure_recovery` cannot help:
  ``except Exception`` does not catch the ``CancelledError`` that kills those
  tasks, and nothing runs at all once the process is gone.

The activity analysis is the one where being stranded was terminal rather than
merely ugly: ``trigger_analysis`` early-returns ``{"status": "pending"}`` for
exactly that state, so before this the affected activity could never be analysed
again, by any route.

Both halves are needed. The sweep catches what a restart stranded but cannot see
a task that dies inside a live process; the age check catches that but only once
the budget has run down.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import select

from backend.app.core.config import settings

log = logging.getLogger(__name__)

#: How long a run may go without visible progress before a reader declares it
#: dead. This is an *inactivity* budget, not a duration budget: every surface
#: touches its timestamp on each progress commit (issue #91), so a healthy run
#: refreshes it every ~500 ms while prose is arriving and on every tool step in
#: between. That is what makes 15 minutes safe where the old 30 was not — the
#: transport's own read timeout (``llm_streaming._STREAM_TIMEOUT``, 300 s
#: between chunks) fails a genuinely silent stream first, so this only has to
#: cover the case where the process died without anyone raising, and it can be
#: tight enough that a stuck run is recoverable in minutes rather than half an
#: hour.
PENDING_TIMEOUT_MINUTES = 15


def _aware(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def pending_timed_out(
    updated_at: Optional[datetime], now: Optional[datetime] = None
) -> bool:
    """Whether a ``pending`` row last touched at ``updated_at`` has run out.

    A ``None`` timestamp counts as timed out: it means either a row written
    before the column existed or a run that never recorded a single step, and in
    both cases there is no evidence anything is still alive. Naive values are
    read as UTC, which is how every writer here stores them.
    """
    updated_at = _aware(updated_at)
    if updated_at is None:
        return True
    now = now or datetime.now(timezone.utc)
    return (now - updated_at).total_seconds() > PENDING_TIMEOUT_MINUTES * 60


# ── Settling one row, per surface ───────────────────────────────────────────
#
# Each of these is "what an unfinishable run should look like once we admit it":
# `error` so the card offers a retry, no leftover progress code under it, and the
# clock touched so the row's age reflects the decision rather than the abandoned
# run. Shared between the routers' age check and the startup sweep so the two
# can't drift.


def settle_training_status(athlete, now: Optional[datetime] = None) -> bool:
    if athlete.training_status_status != "pending":
        return False
    athlete.training_status_status = "error"
    athlete.training_status_progress = None
    athlete.training_status_updated_at = now or datetime.now(timezone.utc)
    return True


def settle_goal_guidance(goal, now: Optional[datetime] = None) -> bool:
    if goal.guidance_status != "pending":
        return False
    goal.guidance_status = "error"
    goal.guidance_updated_at = now or datetime.now(timezone.utc)
    return True


def settle_course_plan(course, now: Optional[datetime] = None) -> bool:
    if course.plan_status != "pending":
        return False
    course.plan_status = "error"
    course.plan_updated_at = now or datetime.now(timezone.utc)
    return True


def settle_activity_analysis(activity, now: Optional[datetime] = None) -> bool:
    if activity.analysis_status != "pending":
        return False
    activity.analysis_status = "error"
    activity.analysis_progress = None
    activity.analysis_updated_at = now or datetime.now(timezone.utc)
    return True


def settle_activity_analysis_if_timed_out(
    activity, now: Optional[datetime] = None
) -> bool:
    """Age out one activity's ``pending`` analysis. Returns True when it did.

    The caller commits — this is called from read paths that may have nothing
    else to write.
    """
    now = now or datetime.now(timezone.utc)
    if activity.analysis_status != "pending":
        return False
    if not pending_timed_out(activity.analysis_updated_at, now):
        return False
    return settle_activity_analysis(activity, now)


# ── The startup sweep ───────────────────────────────────────────────────────


def user_ids_with_a_database() -> list[str]:
    """Every user id that has a database on disk.

    Read from the filesystem rather than the registry on purpose: opening a
    session for a user id that has no ``user.db`` would *create* an empty one,
    and the set that can hold a stranded row is exactly the set of files that
    already exist. Mirrors ``backend/scripts/migrate_user_dbs.py``, which walks
    the same directory for the same reason.
    """
    users_dir = Path(settings.data_dir) / "users"
    if not users_dir.is_dir():
        return []
    return sorted(
        d.name for d in users_dir.iterdir() if d.is_dir() and (d / "user.db").exists()
    )


async def settle_stranded_user_runs(user_id: str, now: Optional[datetime] = None) -> int:
    """Settle every ``pending`` row in one user's database. Returns how many."""
    from backend.app.db.user_session import get_user_session_factory
    from backend.app.models.user_orm import Activity, Athlete, Course, Goal

    now = now or datetime.now(timezone.utc)
    settled = 0
    async with get_user_session_factory(user_id)() as session:
        athletes = (
            await session.execute(
                select(Athlete).where(Athlete.training_status_status == "pending")
            )
        ).scalars().all()
        for athlete in athletes:
            # `training_status_date` is deliberately left alone. The router sets
            # it to today when *it* times a run out, to stop the auto-refresh
            # immediately re-firing a run that just failed; here the run didn't
            # fail, it was killed by the restart, so letting the next read
            # regenerate it is the better outcome — the athlete sees a fresh
            # status instead of an error they have to clear by hand.
            settled += settle_training_status(athlete, now)

        goals = (
            await session.execute(
                select(Goal).where(Goal.guidance_status == "pending")
            )
        ).scalars().all()
        for goal in goals:
            settled += settle_goal_guidance(goal, now)

        activities = (
            await session.execute(
                select(Activity).where(Activity.analysis_status == "pending")
            )
        ).scalars().all()
        for activity in activities:
            settled += settle_activity_analysis(activity, now)

        courses = (
            await session.execute(
                select(Course).where(Course.plan_status == "pending")
            )
        ).scalars().all()
        for course in courses:
            settled += settle_course_plan(course, now)

        if settled:
            await session.commit()
    return settled


async def settle_stranded_runs(now: Optional[datetime] = None) -> int:
    """Settle the ``pending`` rows every user's database was left holding.

    Awaited during ``lifespan`` startup, before the app accepts a request: doing
    it there rather than in a background task means it cannot race a run
    triggered by an early request and settle a row that is genuinely alive.

    One user's failure never stops the sweep — a database that is unreadable
    (mid-migration, say) costs that user their recovery this boot, not everyone
    else's.
    """
    now = now or datetime.now(timezone.utc)
    settled = 0
    for user_id in user_ids_with_a_database():
        try:
            settled += await settle_stranded_user_runs(user_id, now)
        except Exception:
            log.exception("Could not settle stranded LLM runs for user %s", user_id)
    return settled
