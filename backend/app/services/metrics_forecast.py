"""Forward projection of Fitness/Fatigue/Form from planned workouts (issue #34).

``metrics_engine`` runs the Banister model *backwards* over ``Activity.load`` and
stops at today. This module runs the very same pure function
(:func:`openkoutsi.fatigue_metrics.compute_daily_metrics`) *forwards* over
``PlannedWorkout.target_load``, answering "will I be fresh on race day?" and
"is this plan ramping too fast?" before the athlete rides it.

Nothing is persisted. A forecast is only ever as good as the plan it was derived
from, so a cached one would need the same self-healing treatment
``plan_adherence_daily`` needed; recomputing it is a few hundred iterations of a
two-line recurrence, so it is computed on read instead.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.models.user_orm import DailyMetric, TrainingPlan
from backend.app.services.plan_adherence import workout_date
from openkoutsi.fatigue_metrics import compute_daily_metrics

# Horizon defaults/bounds for the projection. A fixed default (rather than "run
# to the end of the last active plan") keeps the chart the same length as a plan
# is consumed, and leaves the decaying tail past the plan's end visible.
DEFAULT_FORECAST_DAYS = 90
MAX_FORECAST_DAYS = 365


async def _seed(
    athlete_id: str, today: date, session: AsyncSession
) -> tuple[float, float, date]:
    """Seed Fitness/Fatigue from the most recent ``DailyMetric`` up to *today*.

    Returns ``(fitness, fatigue, seed_date)``. An athlete with no history seeds
    ``0.0 / 0.0`` at *today*.
    """
    result = await session.execute(
        select(DailyMetric)
        .where(DailyMetric.athlete_id == athlete_id, DailyMetric.date <= today)
        .order_by(DailyMetric.date.desc())
        .limit(1)
    )
    metric = result.scalar_one_or_none()
    if metric is None:
        return 0.0, 0.0, today
    return metric.fitness, metric.fatigue, metric.date


async def planned_load_by_date(
    athlete_id: str, from_date: date, to_date: date, session: AsyncSession
) -> dict[date, float]:
    """Prescribed Load per calendar date over ``[from_date, to_date]``.

    ``PlannedWorkout`` stores ``week_number`` + ``day_of_week`` rather than a
    date, so each workout is placed via :func:`workout_date` relative to its
    plan's ``start_date``. Plans with no ``start_date`` can't be placed on a
    calendar and are skipped; archived plans are excluded by the status filter.

    Creating a plan only archives *overlapping* active plans, so several
    non-overlapping plans can be active at once and two can both contribute
    around a boundary — loads are therefore **summed** across plans rather than
    one plan being picked.

    A workout with no ``target_load`` contributes nothing: the projection never
    invents load it wasn't given. Dates with no entry are treated as ``0.0`` by
    ``compute_daily_metrics``, so rest days decay instead of being skipped.
    """
    result = await session.execute(
        select(TrainingPlan)
        .where(
            TrainingPlan.athlete_id == athlete_id,
            TrainingPlan.status == "active",
        )
        # Only the workouts themselves are needed — unlike plan_adherence, the
        # projection has no use for the linked activities.
        .options(selectinload(TrainingPlan.workouts))
    )

    load_by_date: dict[date, float] = {}
    for plan in result.scalars():
        if plan.start_date is None:
            continue
        for w in plan.workouts:
            if not w.target_load:
                continue
            day = workout_date(plan.start_date, w.week_number, w.day_of_week)
            if from_date <= day <= to_date:
                load_by_date[day] = load_by_date.get(day, 0.0) + float(w.target_load)
    return load_by_date


async def forecast_fitness(
    athlete_id: str,
    session: AsyncSession,
    days: int = DEFAULT_FORECAST_DAYS,
    today: Optional[date] = None,
) -> list[dict]:
    """Project Fitness/Fatigue/Form over the next ``days`` days.

    The series covers ``[today + 1, today + days]`` — today's ``DailyMetric``
    already reflects completed activities, so the measured history stays
    authoritative for everything up to and including today and the two series
    never overlap.

    Projection continues past the end of the plan with zero load rather than
    stopping there: the decaying tail is what detraining looks like, and it
    keeps the chart from ending abruptly mid-taper.

    Returns the same per-day dicts as :func:`compute_daily_metrics`
    (``date``, ``fitness``, ``fatigue``, ``form``, ``load_day``).
    """
    today = today or date.today()
    horizon = today + timedelta(days=days)

    fitness, fatigue, seed_date = await _seed(athlete_id, today, session)
    load_by_date = await planned_load_by_date(
        athlete_id, today + timedelta(days=1), horizon, session
    )

    # Start from the day after the seed rather than the day after today: when the
    # last stored metric predates today (catch-up hasn't run yet), the gap is
    # bridged by zero-load decay instead of being silently skipped.
    rows = compute_daily_metrics(
        load_by_date,
        seed_date + timedelta(days=1),
        horizon,
        fitness,
        fatigue,
    )
    return [row for row in rows if row["date"] > today]
