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
from openkoutsi.plan_adherence import MATCH_THRESHOLD, meets_threshold
from openkoutsi.sport_matching import sports_match


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


async def resolve_planned_workout_for_activity(
    session: AsyncSession,
    athlete_id: str,
    activity: Activity,
) -> Optional[PlannedWorkout]:
    """Resolve the planned workout an activity should be analysed against.

    Prefers an already-linked planned workout (via ``PlannedWorkoutActivity``);
    otherwise falls back to the workout scheduled for the activity's date in an
    active plan — with **no** load/duration threshold, so a session that
    deviated from plan still surfaces its intended workout.

    Unlike :func:`find_and_link_workout` this never writes: it only reads the
    planned workout so the activity analyser can include it as context (issue
    #31). Resolving by date (rather than relying solely on the link) matters
    because on the FIT-upload path analysis is dispatched before auto-linking
    runs, so the link is frequently not yet present when analysis starts.

    Returns the resolved PlannedWorkout, or None if none applies.
    """
    # 1. Already linked → use that workout directly.
    linked_result = await session.execute(
        select(PlannedWorkout)
        .join(
            PlannedWorkoutActivity,
            PlannedWorkoutActivity.planned_workout_id == PlannedWorkout.id,
        )
        .where(PlannedWorkoutActivity.activity_id == activity.id)
        .limit(1)
    )
    linked = linked_result.scalar_one_or_none()
    if linked is not None:
        return linked

    if activity.start_time is None:
        return None

    act_date = (
        activity.start_time.date()
        if hasattr(activity.start_time, "date")
        else activity.start_time
    )
    # isoweekday(): Monday=1, Sunday=7 — matches PlannedWorkout.day_of_week
    day_of_week = act_date.isoweekday()

    plans_result = await session.execute(
        select(TrainingPlan).where(
            TrainingPlan.athlete_id == athlete_id,
            TrainingPlan.status == "active",
        )
    )
    plans = plans_result.scalars().all()

    # Among same-day candidates prefer one whose sport matches the activity;
    # otherwise fall back to the first scheduled workout for the day.
    fallback: Optional[PlannedWorkout] = None
    for plan in plans:
        if plan.start_date is None:
            continue
        if act_date < plan.start_date:
            continue
        if plan.end_date is not None and act_date > plan.end_date:
            continue

        days_elapsed = (act_date - plan.start_date).days
        week_number = days_elapsed // 7 + 1

        workouts_result = await session.execute(
            select(PlannedWorkout).where(
                PlannedWorkout.plan_id == plan.id,
                PlannedWorkout.week_number == week_number,
                PlannedWorkout.day_of_week == day_of_week,
            )
        )
        for workout in workouts_result.scalars().all():
            if sports_match(activity.sport_type, workout.workout_type):
                return workout
            if fallback is None:
                fallback = workout

    return fallback


def _matches(activity: Activity, workout: PlannedWorkout) -> bool:
    if not sports_match(activity.sport_type, workout.workout_type):
        return False

    # Shared with the adherence scoring (openkoutsi.plan_adherence) so the
    # auto-match gate and the per-workout score are defined against the same
    # target-relative comparison and cannot drift apart.
    if not meets_threshold(activity.load, workout.target_load, MATCH_THRESHOLD):
        return False

    planned_duration_s = (workout.duration_min or 0) * 60
    if not meets_threshold(activity.duration_s, planned_duration_s, MATCH_THRESHOLD):
        return False

    return True
