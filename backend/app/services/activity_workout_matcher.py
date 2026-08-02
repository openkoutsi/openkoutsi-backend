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
from openkoutsi.sport_matching import is_rest_workout, sports_match


async def find_and_link_workout(
    session: AsyncSession,
    athlete_id: str,
    activity: Activity,
) -> Optional[PlannedWorkout]:
    """Find a planned workout matching *activity* and link the activity to it.

    Matching rules (all must pass):
    - Activity date falls within the plan's [start_date, end_date]
    - Same week_number and day_of_week relative to the plan's start_date
    - The planned workout is not a rest day
    - Sport type compatible with workout type
    - activity.load >= 60% of planned target_load (when both present)
    - activity.duration_s >= 60% of planned duration_min in seconds (when both present)
    - planned workout does not already have any linked activity
    - the activity is not already linked to some planned workout

    Auto-matching only ever attaches a single activity to an otherwise-empty
    planned workout; additional activities that together complete a workout (for
    example a ride accidentally split in two) are linked manually. Manual links
    have no threshold.

    Returns the linked PlannedWorkout, or None if no match found.
    """
    if activity.start_time is None:
        return None

    # An activity belongs to at most one planned workout (unique constraint on
    # the join table), so re-running the matcher over an already-linked activity
    # — a re-sync, a reprocess, a re-uploaded FIT file — must be a no-op rather
    # than an INSERT that trips the constraint.
    already_linked = await session.execute(
        select(PlannedWorkoutActivity.planned_workout_id)
        .where(PlannedWorkoutActivity.activity_id == activity.id)
        .limit(1)
    )
    if already_linked.scalar_one_or_none() is not None:
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
    activity: Activity,
) -> Optional[PlannedWorkout]:
    """Return the planned workout an activity is explicitly linked to, if any.

    The activity analyser includes this as context so the coach can comment on
    plan adherence (issue #31). We deliberately surface only an *explicit* link
    (via ``PlannedWorkoutActivity``) and never guess a mapping from the
    activity's date: guessing wrongly attributes, for example, an easy commute
    spin to the day's key session that another ride already completed.

    Unlike :func:`find_and_link_workout` this never writes — it only reads the
    linked planned workout.

    Returns the linked PlannedWorkout, or None if the activity is not linked.
    """
    result = await session.execute(
        select(PlannedWorkout)
        .join(
            PlannedWorkoutActivity,
            PlannedWorkoutActivity.planned_workout_id == PlannedWorkout.id,
        )
        .where(PlannedWorkoutActivity.activity_id == activity.id)
        .limit(1)
    )
    return result.scalar_one_or_none()


def _matches(activity: Activity, workout: PlannedWorkout) -> bool:
    # A rest day is a planned *absence* of a session, so nothing may auto-link to
    # it (issue #40). It would otherwise be the loosest target in the plan and win
    # every time: `sports_match` treats "rest" as a generic type that accepts any
    # endurance sport, and its NULL target_load/duration_min make both threshold
    # gates pass unconditionally — so any ride on a rest day was silently
    # swallowed, leaving the athlete unable to link it to the session they
    # actually did and no visible link anywhere to undo.
    if is_rest_workout(workout.workout_type):
        return False

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
