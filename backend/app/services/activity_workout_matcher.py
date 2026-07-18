"""Service to auto-link a processed activity to a matching planned workout."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.user_orm import (
    Activity,
    PlannedWorkout,
    PlannedWorkoutActivity,
    TrainingPlan,
)
from openkoutsi.sport_matching import sports_match

_TSS_THRESHOLD = 0.60
_DURATION_THRESHOLD = 0.60


async def find_and_link_workout(
    session: AsyncSession,
    athlete_id: str,
    activity: Activity,
) -> Optional[PlannedWorkout]:
    """Find a planned workout matching *activity* and link the activity to it.

    Matching rules (all must pass):
    - Activity date falls within the plan's [start_date, end_date]
    - Same week_number and day_of_week relative to the plan's start_date
    - Sport type compatible with workout type
    - activity.load >= 60% of planned target_load (when both present)
    - activity.duration_s >= 60% of planned duration_min in seconds (when both present)
    - planned workout does not already have any linked activity

    Auto-matching only ever attaches a single activity to an otherwise-empty
    planned workout; additional activities that together complete a workout (for
    example a ride accidentally split in two) are linked manually. Manual links
    have no threshold.

    Returns the linked PlannedWorkout, or None if no match found.
    """
    if activity.start_time is None:
        return None

    act_date = (
        activity.start_time.date()
        if hasattr(activity.start_time, "date")
        else activity.start_time
    )
    # isoweekday(): Monday=1, Sunday=7 — matches PlannedWorkout.day_of_week convention
    day_of_week = act_date.isoweekday()

    # Find active plans for this athlete
    plans_result = await session.execute(
        select(TrainingPlan).where(
            TrainingPlan.athlete_id == athlete_id,
            TrainingPlan.status == "active",
        )
    )
    plans = plans_result.scalars().all()
    if not plans:
        return None

    for plan in plans:
        if plan.start_date is None:
            continue
        # Skip plans that haven't started or have ended
        if act_date < plan.start_date:
            continue
        if plan.end_date is not None and act_date > plan.end_date:
            continue

        # Compute the 1-based week number within this plan
        days_elapsed = (act_date - plan.start_date).days
        week_number = days_elapsed // 7 + 1

        # Only consider workouts that have no linked activity yet.
        linked_subq = select(PlannedWorkoutActivity.planned_workout_id).where(
            PlannedWorkoutActivity.planned_workout_id == PlannedWorkout.id
        )
        workouts_result = await session.execute(
            select(PlannedWorkout).where(
                PlannedWorkout.plan_id == plan.id,
                PlannedWorkout.week_number == week_number,
                PlannedWorkout.day_of_week == day_of_week,
                ~linked_subq.exists(),
            )
        )
        candidates = workouts_result.scalars().all()

        for workout in candidates:
            if not _matches(activity, workout):
                continue

            session.add(
                PlannedWorkoutActivity(
                    planned_workout_id=workout.id, activity_id=activity.id
                )
            )
            await session.commit()
            return workout

    return None


def _matches(activity: Activity, workout: PlannedWorkout) -> bool:
    if not sports_match(activity.sport_type, workout.workout_type):
        return False

    if workout.target_load is not None and workout.target_load > 0:
        act_tss = activity.load or 0.0
        if act_tss < workout.target_load * _TSS_THRESHOLD:
            return False

    if workout.duration_min is not None and workout.duration_min > 0:
        planned_duration_s = workout.duration_min * 60
        act_duration_s = activity.duration_s or 0
        if act_duration_s < planned_duration_s * _DURATION_THRESHOLD:
            return False

    return True
