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
* :func:`settle_stranded_runs` — the startup sweep. Nothing that writes a
  ``pending`` status survives its process: the auto-analyse paths run under
  ``asyncio.create_task`` (cancelled at loop shutdown) and ``trigger_analysis``
  under ``BackgroundTasks`` (waited on only up to uvicorn's graceful-shutdown
  timeout). An ordinary redeploy is therefore enough to strand whatever was in
  flight, and :func:`~backend.app.services.llm_streaming.failure_recovery`
  cannot help: ``except Exception`` does not catch the ``CancelledError`` that
  kills those tasks, and nothing runs at all once the process is gone.

  The sweep used to settle **every** ``pending`` row it found, on the premise
  that a row in that state at boot belonged to a process that was gone. That
  premise holds only while exactly one process exists (issue #50). Behind a
  proxy a rolling redeploy overlaps two, and the one booting would settle the
  live runs of the one still serving — the worst regression a naive scale-out
  would introduce, and reachable today without any replicas at all.

  So the sweep asks the same question the routers ask: has the heartbeat run
  down? A row still being written to belongs to *somebody*, and this process not
  knowing who is not evidence that nobody does. The cost is that a run killed by
  a restart now waits out the remainder of its budget instead of being released
  at boot — bounded by ``PENDING_TIMEOUT_MINUTES``, and settled by the read that
  discovers it rather than left stuck, because every surface carries that check
  on its read path.

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
import uuid
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


# ── Run ownership ───────────────────────────────────────────────────────────


async def run_is_current(session, model, pk, token_column, run_id) -> bool:
    """Does the row still belong to the run holding ``run_id``?

    A fresh ``SELECT``, not the loaded instance: the point is to see what
    another session committed while this one was streaming.

    ``run_id is None`` keeps the old behaviour, so a run already in flight
    across the upgrade that added the token is never discarded.

    The complement of :func:`pending_timed_out` — that says whether a run is
    *alive*, this says whether its writes are still *wanted*.
    """
    if run_id is None:
        return True
    current = (
        await session.execute(
            select(token_column).where(model.id == pk)
        )
    ).scalar_one_or_none()
    return current == run_id


# ── Starting one run, per surface ───────────────────────────────────────────
#
# The mirror of `settle_*` below: what a row looks like the moment a run claims
# it. Each mints the token the run owns its columns by and returns it for the
# caller to pass to the background task. Kept adjacent so the columns one
# touches cannot drift from the columns the other clears.


def begin_training_status_run(athlete, now: Optional[datetime] = None) -> str:
    run_id = uuid.uuid4().hex
    athlete.training_status_status = "pending"
    athlete.training_status = None
    athlete.training_status_progress = None
    athlete.training_status_run_id = run_id
    athlete.training_status_updated_at = now or datetime.now(timezone.utc)
    return run_id


def begin_goal_guidance_run(goal, now: Optional[datetime] = None) -> str:
    run_id = uuid.uuid4().hex
    goal.guidance_status = "pending"
    goal.guidance = None
    goal.guidance_verdict = None
    goal.guidance_run_id = run_id
    goal.guidance_updated_at = now or datetime.now(timezone.utc)
    return run_id


def begin_activity_analysis_run(activity, now: Optional[datetime] = None) -> str:
    run_id = uuid.uuid4().hex
    activity.analysis_status = "pending"
    # Cleared here rather than by the caller, like both siblings above: in
    # `trigger_analysis` the invariant held only for callers that remembered
    # it, leaving `pending` on top of the previous run's prose.
    activity.analysis = None
    activity.analysis_progress = None
    activity.analysis_run_id = run_id
    activity.analysis_updated_at = now or datetime.now(timezone.utc)
    return run_id


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
    # Retire the run token: a run declared dead must not be able to come back
    # and overwrite the settled state if it was merely slow.
    athlete.training_status_run_id = None
    athlete.training_status_updated_at = now or datetime.now(timezone.utc)
    return True


def settle_goal_guidance(goal, now: Optional[datetime] = None) -> bool:
    if goal.guidance_status != "pending":
        return False
    goal.guidance_status = "error"
    goal.guidance_run_id = None
    goal.guidance_updated_at = now or datetime.now(timezone.utc)
    return True


def settle_course_surface(course, now: Optional[datetime] = None) -> bool:
    """Settle a stranded surface match. Delegates to the module that owns it.

    Imported lazily: ``course_surface`` reaches the matcher and the analysis
    service, and this module is imported during startup before any of that is
    needed.
    """
    from backend.app.services.course_surface import settle_course_surface as _settle

    return _settle(course, now)


def settle_course_plan(course, now: Optional[datetime] = None) -> bool:
    if course.plan_status != "pending":
        return False
    course.plan_status = "error"
    # Retire the run token as well: a run declared dead must not be able to
    # come back and overwrite the settled state if it was merely slow.
    course.plan_run_id = None
    course.plan_updated_at = now or datetime.now(timezone.utc)
    return True


def settle_activity_analysis(activity, now: Optional[datetime] = None) -> bool:
    if activity.analysis_status != "pending":
        return False
    activity.analysis_status = "error"
    activity.analysis_progress = None
    activity.analysis_run_id = None
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
    """Settle the timed-out ``pending`` rows in one user's database.

    Returns how many. The age check runs in Python rather than in the ``WHERE``
    clause: SQLite hands these timestamps back naive, which is why
    :func:`_aware` exists, and a comparison against an aware ``now`` pushed into
    SQL would not survive that. The row counts here are small enough that it
    does not matter.
    """
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
            if not pending_timed_out(athlete.training_status_updated_at, now):
                continue
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
            if not pending_timed_out(goal.guidance_updated_at, now):
                continue
            settled += settle_goal_guidance(goal, now)

        activities = (
            await session.execute(
                select(Activity).where(Activity.analysis_status == "pending")
            )
        ).scalars().all()
        for activity in activities:
            if not pending_timed_out(activity.analysis_updated_at, now):
                continue
            settled += settle_activity_analysis(activity, now)

        courses = (
            await session.execute(
                select(Course).where(Course.plan_status == "pending")
            )
        ).scalars().all()
        for course in courses:
            if not pending_timed_out(course.plan_updated_at, now):
                continue
            settled += settle_course_plan(course, now)

        # A surface match a redeploy interrupted (issue #56). Same shape and
        # same reasoning as the plan above, but it settles to `unavailable`
        # rather than `error`: nothing about the course is wrong, the match
        # simply did not finish, and the athlete still has a complete Stage 1
        # result in front of them.
        matching = (
            await session.execute(
                select(Course).where(Course.surface_status == "pending")
            )
        ).scalars().all()
        for course in matching:
            if not pending_timed_out(course.surface_updated_at, now):
                continue
            settled += settle_course_surface(course, now)

        if settled:
            await session.commit()
    return settled


async def settle_stranded_runs(now: Optional[datetime] = None) -> int:
    """Settle the ``pending`` rows every user's database was left holding.

    Only the ones whose heartbeat has run down: see the module docstring for why
    "there is a ``pending`` row and we just booted" is not on its own evidence
    that the run behind it is dead.

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
