"""Closing a training plan once it has run its course.

``status`` used to have no way out of ``active``: ``archived`` is set when an
overlapping new plan supersedes a plan, or when the athlete files it away by
hand, and neither happens on its own. A plan whose eight weeks ended in March
was therefore still ``active`` in September — still drawn as the current plan,
still handed to the coach as what the athlete is following, and still accruing
one identical ``plan_adherence_daily`` row a day.

So a plan closes itself: once its last scheduled day has passed it moves to
``completed``. The rule is deliberately the one
``achievements._compute_plan_rules`` already applies — a plan counts as finished
when its end date is behind us — so the "Plan finisher" badge and the plan's own
status can never disagree about whether it is over.

Two things it is careful about:

- **Whose day it is.** The boundary is the athlete's local date, resolved the
  same way the achievement rules resolve it. A plan ending on a Sunday should
  close when that Sunday is over where the athlete lives, not where the server
  is.
- **Not fighting the athlete.** Reopening a finished plan leaves ``completed_at``
  set, and this pass only closes plans that have never been closed. Without that
  a reopened plan would be shut again by the very next read. Moving the plan's
  dates clears the timestamp (see ``api.plans``), which is what lets a re-dated
  plan finish again on its new end date.

Deterministic and idempotent, so it can run wherever it is cheapest to notice:
:func:`plan_adherence.catch_up_adherence` calls it before scoring, which covers
every ingest path and the daily first read, and the plan endpoints call it
directly so the plan page is never the stale one.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.timezones import resolve_zone
from backend.app.models.user_orm import Athlete, TrainingPlan

log = logging.getLogger(__name__)

#: Statuses that describe a plan still in play. ``archived`` is excluded: a plan
#: the athlete filed away is not waiting to be closed. Exported because the
#: consumers that used to ask for ``status == "active"`` — the activity matcher,
#: the adherence snapshots, the forecast — mean "not filed away" rather than
#: "not yet finished", and a finished plan is still theirs to read.
LIVE_STATUSES = ("active", "completed")


async def _resolve_today(
    athlete_id: str, session: AsyncSession, athlete: Optional[Athlete]
) -> date:
    """The athlete's current local date, falling back to UTC."""
    if athlete is None:
        athlete = (
            await session.execute(select(Athlete).where(Athlete.id == athlete_id))
        ).scalar_one_or_none()
    tz = resolve_zone((athlete.app_settings or {}).get("timezone")) if athlete else None
    return datetime.now(tz or timezone.utc).date()


async def close_finished_plans(
    athlete_id: str,
    session: AsyncSession,
    *,
    athlete: Optional[Athlete] = None,
    today: Optional[date] = None,
) -> list[TrainingPlan]:
    """Close every plan of this athlete's whose last day has passed.

    Returns the plans closed by *this* call — empty on the common path, which is
    what makes it cheap enough to run on a read. ``athlete`` and ``today`` are
    both optional short-cuts for callers that already hold them; ``today`` given
    explicitly wins, and is how the tests pin the boundary.

    A plan with no ``end_date`` is never closed: an open-ended plan has no last
    day to be past, and guessing one from ``weeks`` would close plans whose
    duration was never recorded.
    """
    if today is None:
        today = await _resolve_today(athlete_id, session, athlete)

    result = await session.execute(
        select(TrainingPlan).where(
            TrainingPlan.athlete_id == athlete_id,
            TrainingPlan.status == "active",
            TrainingPlan.completed_at.is_(None),
            TrainingPlan.end_date.is_not(None),
            TrainingPlan.end_date < today,
        )
    )
    closed = list(result.scalars().all())
    if not closed:
        return []

    now = datetime.now(timezone.utc)
    for plan in closed:
        plan.status = "completed"
        plan.completed_at = now
    await session.commit()

    log.info(
        "Closed %d finished training plan(s) for athlete %s", len(closed), athlete_id
    )
    return closed
