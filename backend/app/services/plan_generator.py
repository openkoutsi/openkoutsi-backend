"""Training plan persistence — creates TrainingPlan and PlannedWorkout rows."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.team_orm import TrainingPlan, PlannedWorkout
from openkoutsi.plan_schema import PlanConfig
from openkoutsi.plan_builder import week_template, build_week_from_config


def build_workout_rows(
    plan_id: str,
    num_weeks: int,
    goal: Optional[str],
    config: Optional[PlanConfig] = None,
) -> list[PlannedWorkout]:
    """Build (unsaved) PlannedWorkout rows for a plan from a template/config."""
    workouts: list[PlannedWorkout] = []
    for week_num in range(1, num_weeks + 1):
        if config is not None:
            template = build_week_from_config(config, week_num, num_weeks)
        else:
            template = week_template(week_num, num_weeks, goal)

        for day in template:
            workouts.append(PlannedWorkout(plan_id=plan_id, week_number=week_num, **day))
    return workouts


async def generate_plan(
    athlete_id: str,
    name: str,
    start_date: date,
    num_weeks: int,
    goal: Optional[str],
    session: AsyncSession,
    config: Optional[PlanConfig] = None,
) -> TrainingPlan:
    """Create a TrainingPlan with PlannedWorkout rows."""

    end_date = start_date + timedelta(weeks=num_weeks) - timedelta(days=1)

    plan = TrainingPlan(
        athlete_id=athlete_id,
        name=name,
        start_date=start_date,
        end_date=end_date,
        goal=goal,
        weeks=num_weeks,
        status="active",
        config=config.model_dump() if config else None,
        generation_method="rule_based",
    )
    session.add(plan)
    await session.flush()

    session.add_all(build_workout_rows(plan.id, num_weeks, goal, config))
    await session.commit()
    await session.refresh(plan)
    return plan
