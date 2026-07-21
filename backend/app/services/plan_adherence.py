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
    COMPLETED_MIN_SCORE,
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
    activities; supplemental workouts are done/missed. The result is floored at
    ``COMPLETED_MIN_SCORE`` — having actually done the session, however far off
    target, always beats missing it outright.
    """
    if workout_is_cycling(workout.workout_type):
        actual_load = sum((a.load or 0.0) for a in workout.linked_activities)
        actual_dur = sum((a.duration_s or 0) for a in workout.linked_activities)
        raw = cycling_match_score(
            actual_load, actual_dur, workout.target_load, workout.duration_min
        )
    else:
        raw = supplemental_match_score(True)
    return max(COMPLETED_MIN_SCORE, raw)


@dataclass
class PlanScore:
    """Result of scoring a plan as of a given day."""

    score: Optional[float]  # 0–100 "so far" adherence, or None (nothing yet)
    completed: int = 0
    missed: int = 0
    skipped: int = 0
    pending: int = 0
    future: int = 0  # non-rest workouts still ahead (dated after today)
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

        # Future workouts are excluded from the "so far" denominator, but are
        # still counted as remaining sessions the athlete has yet to do.
        if wdate > today:
            result.match_scores[w.id] = None
            result.future += 1
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
        # Sessions run with expire_on_commit=False, so plans/workouts already
        # loaded in this session could carry stale collections. Recompute must
        # see current DB state (self-healing depends on it), so overwrite any
        # already-loaded attributes with a fresh read.
        .execution_options(populate_existing=True)
    )
    return list(result.scalars().all())


def _apply_snapshot(row: PlanAdherenceDaily, ps: PlanScore) -> None:
    row.score = ps.score
    row.completed = ps.completed
    row.missed = ps.missed
    row.skipped = ps.skipped
    row.pending = ps.pending


def _snapshot_differs(row: PlanAdherenceDaily, ps: PlanScore) -> bool:
    """Whether a stored snapshot no longer matches a freshly computed score."""
    if (row.score is None) != (ps.score is None):
        return True
    if row.score is not None and ps.score is not None and abs(row.score - ps.score) > 0.01:
        return True
    return (
        row.completed != ps.completed
        or row.missed != ps.missed
        or row.skipped != ps.skipped
        or row.pending != ps.pending
    )


async def catch_up_adherence(athlete_id: str, session: AsyncSession) -> bool:
    """Recompute the ``plan_adherence_daily`` snapshot series for active plans.

    For each active plan, every day in ``[start_date, today]`` is scored "as of"
    that day and compared against what's stored. A day is (re)written when it is
    **missing** or **stale** — the latter self-heals rows invalidated by
    retroactive changes (an activity linked/unlinked to an old workout, a past
    workout edited or its skip reason changed, or a formula change), the same way
    ``metrics_engine.catch_up_metrics`` heals stale ``daily_metrics``. Days that
    already match are left untouched, so the pass is deterministic and
    idempotent. Returns True if any row was written.
    """
    today = date.today()
    plans = await _load_active_plans(athlete_id, session)
    if not plans:
        return False

    changed = False
    for plan in plans:
        if plan.start_date is None or plan.start_date > today:
            continue

        existing = {
            row.date: row
            for row in (
                await session.execute(
                    select(PlanAdherenceDaily).where(
                        PlanAdherenceDaily.athlete_id == athlete_id,
                        PlanAdherenceDaily.plan_id == plan.id,
                    )
                )
            ).scalars()
        }

        day = plan.start_date
        while day <= today:
            ps = score_plan(plan, day)
            row = existing.get(day)
            if row is None:
                row = PlanAdherenceDaily(
                    athlete_id=athlete_id, plan_id=plan.id, date=day
                )
                _apply_snapshot(row, ps)
                session.add(row)
                changed = True
            elif _snapshot_differs(row, ps):
                _apply_snapshot(row, ps)
                changed = True
            day += timedelta(days=1)

    if changed:
        await session.commit()
    return changed
