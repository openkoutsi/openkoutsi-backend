"""Plan-adherence scoring service (issue #26).

Deterministic, always-on (never gated behind the LLM subscription). Wraps the
pure math in :mod:`openkoutsi.plan_adherence` with the DB orchestration:

- derive the per-workout ``match_score`` surfaced in the API,
- roll the per-workout scores up into the "so far" plan adherence score,
- persist a daily snapshot per active plan for charting, with a ``catch_up``
  backfill mirroring ``metrics_engine.catch_up_metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.models.user_orm import PlanAdherenceDaily, PlannedWorkout, TrainingPlan
from openkoutsi.plan_adherence import (
    SUPPLEMENTAL_WEIGHT_FALLBACK,
    cycling_match_score,
    forgiveness_factor,
    plan_adherence,
    supplemental_match_score,
    supplemental_weight,
)
from openkoutsi.sport_matching import workout_is_cycling


def _workout_date(start_date: date, week_number: int, day_of_week: int) -> date:
    """Map a planned workout's (week, day) to an absolute calendar date."""
    return start_date + timedelta(days=(week_number - 1) * 7 + (day_of_week - 1))


def _is_rest(workout: PlannedWorkout) -> bool:
    wtype = (workout.workout_type or "").strip().lower()
    return wtype in ("", "rest")


def _cycling_weight(workout: PlannedWorkout, supp_weight: float) -> float:
    """Weight for a cycling workout: target_load, else duration_min, else flat."""
    if workout.target_load:
        return float(workout.target_load)
    if workout.duration_min:
        return float(workout.duration_min)
    return supp_weight


def workout_match_score(workout: PlannedWorkout) -> float:
    """Per-workout match score (0–100) for a *completed* workout.

    Cycling workouts are graded on summed Load + duration across all linked
    activities; supplemental workouts are done/missed.
    """
    if workout_is_cycling(workout.workout_type):
        actual_load = sum((a.load or 0.0) for a in workout.linked_activities)
        actual_dur = sum((a.duration_s or 0) for a in workout.linked_activities)
        return cycling_match_score(
            actual_load, actual_dur, workout.target_load, workout.duration_min
        )
    return supplemental_match_score(True)


@dataclass
class PlanScore:
    """Result of scoring a plan as of a given day."""

    score: Optional[float]  # 0–100 "so far" adherence, or None (nothing yet)
    completed: int = 0
    missed: int = 0
    skipped: int = 0
    pending: int = 0
    # Per-workout match score (0–100) keyed by workout id; None when not yet
    # scorable (rest day, future, or an empty today workout still in grace).
    match_scores: dict[str, Optional[float]] = field(default_factory=dict)


def score_plan(plan: TrainingPlan, today: date) -> PlanScore:
    """Score *plan* as of *today* from its already-loaded workouts/activities."""
    result = PlanScore(score=None)
    if plan.start_date is None:
        return result

    supp_weight = supplemental_weight(
        w.target_load
        for w in plan.workouts
        if workout_is_cycling(w.workout_type) and w.target_load
    )
    if not supp_weight:
        supp_weight = SUPPLEMENTAL_WEIGHT_FALLBACK

    contributions: list[tuple[float, float]] = []

    for w in plan.workouts:
        # Rest days / no-target placeholders are excluded entirely.
        if _is_rest(w):
            result.match_scores[w.id] = None
            continue

        wdate = _workout_date(plan.start_date, w.week_number, w.day_of_week)

        # Future workouts are excluded from the "so far" denominator.
        if wdate > today:
            result.match_scores[w.id] = None
            continue

        is_cycling = workout_is_cycling(w.workout_type)
        weight = _cycling_weight(w, supp_weight) if is_cycling else supp_weight

        if w.linked_activities:
            # Completed (or a today workout scored provisionally on what's linked).
            ms = workout_match_score(w)
            result.match_scores[w.id] = ms
            contributions.append((weight, ms))
            result.completed += 1
        elif w.skip_reason:
            # Skipped: score 0 at a reason-graded partial weight.
            f = forgiveness_factor(w.skip_reason)
            result.match_scores[w.id] = 0.0
            contributions.append(((1.0 - f) * weight, 0.0))
            result.skipped += 1
        elif wdate == today:
            # Today, still empty: held in grace, not yet a miss.
            result.match_scores[w.id] = None
            result.pending += 1
        else:
            # Past, empty, not skipped, not a rest day: a miss (0 at full weight).
            result.match_scores[w.id] = 0.0
            contributions.append((weight, 0.0))
            result.missed += 1

    result.score = plan_adherence(contributions)
    return result


async def _load_active_plans(
    athlete_id: str, session: AsyncSession
) -> list[TrainingPlan]:
    result = await session.execute(
        select(TrainingPlan)
        .where(
            TrainingPlan.athlete_id == athlete_id,
            TrainingPlan.status == "active",
        )
        .options(
            selectinload(TrainingPlan.workouts).selectinload(
                PlannedWorkout.linked_activities
            )
        )
    )
    return list(result.scalars().all())


async def _upsert_snapshot(
    athlete_id: str, plan_id: str, day: date, ps: PlanScore, session: AsyncSession
) -> None:
    existing = await session.execute(
        select(PlanAdherenceDaily).where(
            PlanAdherenceDaily.athlete_id == athlete_id,
            PlanAdherenceDaily.plan_id == plan_id,
            PlanAdherenceDaily.date == day,
        )
    )
    row = existing.scalar_one_or_none()
    if row is None:
        row = PlanAdherenceDaily(athlete_id=athlete_id, plan_id=plan_id, date=day)
        session.add(row)
    row.score = ps.score
    row.completed = ps.completed
    row.missed = ps.missed
    row.skipped = ps.skipped
    row.pending = ps.pending


async def catch_up_adherence(athlete_id: str, session: AsyncSession) -> bool:
    """Fill missing ``plan_adherence_daily`` rows up to today for active plans.

    For each active plan, snapshots are (re)computed for today (so the score
    moves the moment an activity lands) plus any missing days back to the plan
    start, computing each day's score "as of" that day. Deterministic and
    idempotent — recomputing a day yields the same value. Returns True if any
    row was written.
    """
    today = date.today()
    plans = await _load_active_plans(athlete_id, session)
    if not plans:
        return False

    changed = False
    for plan in plans:
        if plan.start_date is None or plan.start_date > today:
            continue
        start = plan.start_date

        existing_dates = {
            d
            for (d,) in (
                await session.execute(
                    select(PlanAdherenceDaily.date).where(
                        PlanAdherenceDaily.athlete_id == athlete_id,
                        PlanAdherenceDaily.plan_id == plan.id,
                    )
                )
            ).all()
        }

        # Days to compute: any missing day in [start, today] plus today itself
        # (today is always refreshed so newly-linked activities move the score).
        day = start
        while day <= today:
            if day == today or day not in existing_dates:
                ps = score_plan(plan, day)
                await _upsert_snapshot(athlete_id, plan.id, day, ps, session)
                changed = True
            day += timedelta(days=1)

    if changed:
        await session.commit()
    return changed
