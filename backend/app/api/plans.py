from datetime import date, datetime, timedelta, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.deps import get_ctx_and_session
from backend.app.db.registry import get_registry_session
from backend.app.models.registry_orm import InstanceSettings
from backend.app.models.user_orm import (
    Athlete, TrainingPlan, PlannedWorkout, PlannedWorkoutActivity, WorkoutDefinition,
)
from backend.app.models.user_orm import Activity
from backend.app.schemas.plans import (
    TrainingPlanCreate, TrainingPlanUpdate, TrainingPlanResponse,
    LinkActivityRequest, PlannedWorkoutResponse, SkipWorkoutRequest,
    PlannedWorkoutCreate, PlannedWorkoutUpdate, RegeneratePlanRequest,
    GenerateUpcomingWorkoutsRequest, GenerateUpcomingWorkoutsResponse,
    GenerateUpcomingResultItem, PlanAdherenceSummary, PlanAdherencePoint,
)
from backend.app.models.user_orm import PlanAdherenceDaily
from backend.app.services.plan_adherence import (
    score_plan, catch_up_adherence, workout_match_score,
)
from backend.app.services.plan_generator import (
    generate_plan, build_workout_rows, week_meta_for,
)
from openkoutsi.plan_schema import clamp_plan_params
from openkoutsi.plan_builder import week_meta_from_weeks


def _llm_weeks_to_day_dicts(llm_weeks) -> list[list[dict]]:
    """Flatten posted, pre-built LLM weeks into the day dicts week_meta needs.

    Only ``duration_min``/``target_load`` are read by the week summary, but the
    other fields are carried through so the shape matches the generator output.
    """
    return [
        [
            {
                "day_of_week": d.day_of_week,
                "workout_type": d.workout_type,
                "description": d.description,
                "duration_min": d.duration_min,
                "target_load": d.target_load,
            }
            for d in week
        ]
        for week in llm_weeks
    ]
from backend.app.services.llm_plan_generator import generate_plan_llm, generate_plan_weeks_llm
from backend.app.services.llm_workout_generator import (
    generate_workout_definition_llm, WorkoutGenerationError,
)
from backend.app.services.llm_client import LLM_ERROR_STATUS, LlmConfigError
from backend.app.services.llm_access import check_llm_access, subscription_required_error
from backend.app.schemas.pagination import Page, PageParams, paginate_params

router = APIRouter(prefix="/plans", tags=["plans"])

# Workouts are generated for the upcoming week (today → +6 days) by default.
_GENERATE_WINDOW_DAYS = 6


def _planned_date(start_date, week_number: int, day_of_week: int):
    """Map a planned workout's (week, day) to an absolute calendar date."""
    return start_date + timedelta(days=(week_number - 1) * 7 + (day_of_week - 1))


def _plan_response_with_adherence(plan: TrainingPlan) -> TrainingPlanResponse:
    """Build a TrainingPlanResponse decorated with the derived adherence scores.

    Requires ``plan.workouts`` (with their linked activities) to be loaded, which
    the plan endpoints already do. Deterministic and cheap — computed live for
    the response; the persisted daily snapshot is written separately.
    """
    response = TrainingPlanResponse.model_validate(plan)
    ps = score_plan(plan, date.today())
    response.adherence_score = ps.score
    response.adherence_summary = PlanAdherenceSummary(
        completed=ps.completed, missed=ps.missed,
        skipped=ps.skipped, pending=ps.pending,
        remaining=ps.future + ps.pending,
    )
    for workout in response.workouts:
        workout.match_score = ps.match_scores.get(workout.id)
    return response


def _plans_overlap(a_start, a_end, b_start, b_end) -> bool:
    """Whether two plan date ranges overlap.

    Ranges are inclusive [start, end]. If any endpoint is unknown (None) we
    treat the ranges as overlapping, so a plan with incomplete dates is still
    archived when a new one is created (the conservative, pre-existing
    behaviour).
    """
    if a_start is None or a_end is None or b_start is None or b_end is None:
        return True
    return a_start <= b_end and b_start <= a_end


def _plan_end_date(start_date, weeks):
    """Inclusive end date for a plan of ``weeks`` weeks starting on ``start_date``."""
    if start_date is None or not weeks:
        return None
    return start_date + timedelta(weeks=weeks) - timedelta(days=1)


async def _archive_overlapping_active_plans(session, athlete_id, start_date, end_date):
    """Archive active plans whose date range overlaps [start_date, end_date].

    Non-overlapping active plans are left active, so several plans covering
    different time periods can coexist.
    """
    result = await session.execute(
        select(TrainingPlan)
        .where(TrainingPlan.athlete_id == athlete_id, TrainingPlan.status == "active")
    )
    for old in result.scalars().all():
        if _plans_overlap(old.start_date, old.end_date, start_date, end_date):
            old.status = "archived"
    await session.flush()


async def _get_athlete(global_user_id: str, session: AsyncSession) -> Athlete:
    result = await session.execute(select(Athlete).where(Athlete.global_user_id == global_user_id))
    athlete = result.scalar_one_or_none()
    if not athlete:
        raise HTTPException(404, "Athlete profile not found")
    return athlete


@router.get("", response_model=Page[TrainingPlanResponse],
            operation_id="listPlans", summary="List training plans")
async def list_plans(
    ctx_session=Depends(get_ctx_and_session),
    params: PageParams = Depends(paginate_params),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    total = (await session.execute(
        select(func.count()).select_from(TrainingPlan).where(TrainingPlan.athlete_id == athlete.id)
    )).scalar_one()
    result = await session.execute(
        select(TrainingPlan)
        .where(TrainingPlan.athlete_id == athlete.id)
        .options(selectinload(TrainingPlan.workouts))
        .order_by(TrainingPlan.created_at.desc())
        .offset(params.offset)
        .limit(params.page_size)
    )
    items = [_plan_response_with_adherence(p) for p in result.scalars().all()]
    return Page.build(items, total, params.page, params.page_size)


@router.post("", response_model=TrainingPlanResponse, status_code=201)
async def create_plan(
    body: TrainingPlanCreate,
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    ctx, session = ctx_session

    athlete = await _get_athlete(ctx.user_id, session)

    # Clamp structure parameters (progression %, cadence, hours band, …) to sane
    # bounds before generation and persistence.
    config = clamp_plan_params(body.config) if body.config else None

    # Archive existing active plans, but only those whose date range overlaps
    # the new plan — a new plan covering a different period leaves earlier
    # plans active.
    new_start = body.start_date
    new_end = _plan_end_date(body.start_date, body.weeks)
    await _archive_overlapping_active_plans(session, athlete.id, new_start, new_end)

    if body.llm_weeks:
        # Frontend already called the LLM — persist the pre-built weeks directly.
        end_date = body.start_date + timedelta(weeks=body.weeks) - timedelta(days=1)
        # Summarise week_meta from the actual LLM weeks (not a rule-based rebuild),
        # so the reported Load/hours match the workouts the athlete sees.
        llm_week_meta = (
            week_meta_from_weeks(config, _llm_weeks_to_day_dicts(body.llm_weeks))
            if config
            else None
        )
        plan = TrainingPlan(
            athlete_id=athlete.id,
            name=body.name,
            start_date=body.start_date,
            end_date=end_date,
            goal=body.goal,
            weeks=body.weeks,
            status="active",
            config=config.model_dump() if config else None,
            generation_method="llm",
            week_meta=llm_week_meta,
        )
        session.add(plan)
        await session.flush()

        for week_num, week_days in enumerate(body.llm_weeks, start=1):
            for day in week_days:
                session.add(PlannedWorkout(
                    plan_id=plan.id,
                    week_number=week_num,
                    day_of_week=day.day_of_week,
                    workout_type=day.workout_type,
                    description=day.description,
                    duration_min=day.duration_min,
                    target_load=day.target_load,
                ))
        await session.commit()
        await session.refresh(plan)
    elif body.use_llm:
        if not config:
            raise HTTPException(400, "A plan config (training days and types) is required for LLM generation")
        instance_result = await registry_session.execute(select(InstanceSettings).limit(1))
        instance = instance_result.scalar_one_or_none()
        access = await check_llm_access(ctx, athlete, instance, registry_session)
        if not access.allowed:
            raise subscription_required_error()
        try:
            plan = await generate_plan_llm(
                athlete=athlete,
                config=config,
                name=body.name,
                start_date=body.start_date,
                num_weeks=body.weeks,
                goal=body.goal,
                session=session,
                instance=instance,
                user_id=ctx.user_id,
                allow_instance_fallback=(access.mode != "byok"),
            )
        except LlmConfigError as exc:
            raise HTTPException(LLM_ERROR_STATUS.get(exc.code, 400), str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(503, f"LLM plan generation failed: {exc}") from exc
    else:
        plan = await generate_plan(
            athlete_id=athlete.id,
            name=body.name,
            start_date=body.start_date,
            num_weeks=body.weeks,
            goal=body.goal,
            session=session,
            config=config,
        )

    # Reload with workouts
    result = await session.execute(
        select(TrainingPlan)
        .where(TrainingPlan.id == plan.id)
        .options(selectinload(TrainingPlan.workouts))
    )
    plan = result.scalar_one()
    return _plan_response_with_adherence(plan)


@router.get("/{plan_id}", response_model=TrainingPlanResponse)
async def get_plan(
    plan_id: str,
    ctx_session=Depends(get_ctx_and_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    result = await session.execute(
        select(TrainingPlan)
        .where(TrainingPlan.id == plan_id, TrainingPlan.athlete_id == athlete.id)
        .options(selectinload(TrainingPlan.workouts))
    )
    plan = result.scalar_one_or_none()
    if not plan:
        raise HTTPException(404, "Plan not found")
    return _plan_response_with_adherence(plan)


@router.get("/{plan_id}/adherence", response_model=list[PlanAdherencePoint],
            operation_id="getPlanAdherence", summary="Plan adherence snapshot series")
async def get_plan_adherence(
    plan_id: str,
    start: Optional[date] = None,
    end: Optional[date] = None,
    ctx_session=Depends(get_ctx_and_session),
):
    """Persisted daily adherence snapshots for a plan (mirrors /metrics/fitness).

    Runs a cheap, deterministic catch-up first so the series is fresh even for a
    user who hasn't loaded the dashboard yet, then returns the stored trend.
    """
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    plan_result = await session.execute(
        select(TrainingPlan).where(
            TrainingPlan.id == plan_id, TrainingPlan.athlete_id == athlete.id
        )
    )
    if not plan_result.scalar_one_or_none():
        raise HTTPException(404, "Plan not found")

    await catch_up_adherence(athlete.id, session)

    query = select(PlanAdherenceDaily).where(
        PlanAdherenceDaily.athlete_id == athlete.id,
        PlanAdherenceDaily.plan_id == plan_id,
    )
    if start:
        query = query.where(PlanAdherenceDaily.date >= start)
    if end:
        query = query.where(PlanAdherenceDaily.date <= end)
    result = await session.execute(query.order_by(PlanAdherenceDaily.date))
    return [PlanAdherencePoint.model_validate(r) for r in result.scalars().all()]


@router.put("/{plan_id}", response_model=TrainingPlanResponse)
async def update_plan(
    plan_id: str,
    body: TrainingPlanUpdate,
    ctx_session=Depends(get_ctx_and_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    result = await session.execute(
        select(TrainingPlan)
        .where(TrainingPlan.id == plan_id, TrainingPlan.athlete_id == athlete.id)
        .options(selectinload(TrainingPlan.workouts))
    )
    plan = result.scalar_one_or_none()
    if not plan:
        raise HTTPException(404, "Plan not found")

    if body.status is not None:
        plan.status = body.status
    if body.name is not None:
        plan.name = body.name
    if body.goal is not None:
        plan.goal = body.goal
    if body.start_date is not None:
        plan.start_date = body.start_date
    if body.weeks is not None:
        plan.weeks = body.weeks

    # Recompute end_date when the start date or duration changed.
    if (body.start_date is not None or body.weeks is not None) and plan.start_date and plan.weeks:
        plan.end_date = plan.start_date + timedelta(weeks=plan.weeks) - timedelta(days=1)

    await session.commit()
    await session.refresh(plan)
    return _plan_response_with_adherence(plan)


@router.post("/{plan_id}/unarchive", response_model=TrainingPlanResponse,
             operation_id="unarchivePlan", summary="Unarchive a training plan")
async def unarchive_plan(
    plan_id: str,
    ctx_session=Depends(get_ctx_and_session),
):
    """Reactivate an archived plan.

    Any currently-active plan whose date range overlaps the reactivated plan is
    archived, so overlapping plans are never both active at once. Active plans
    covering a different period are left untouched.
    """
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    result = await session.execute(
        select(TrainingPlan)
        .where(TrainingPlan.id == plan_id, TrainingPlan.athlete_id == athlete.id)
        .options(selectinload(TrainingPlan.workouts))
    )
    plan = result.scalar_one_or_none()
    if not plan:
        raise HTTPException(404, "Plan not found")

    if plan.status != "active":
        await _archive_overlapping_active_plans(
            session, athlete.id, plan.start_date, plan.end_date
        )
        plan.status = "active"

    await session.commit()
    await session.refresh(plan)
    result = await session.execute(
        select(TrainingPlan)
        .where(TrainingPlan.id == plan.id)
        .options(selectinload(TrainingPlan.workouts))
    )
    plan = result.scalar_one()
    return _plan_response_with_adherence(plan)


@router.put("/{plan_id}/workouts/{workout_id}/link", response_model=PlannedWorkoutResponse)
async def link_workout_to_activity(
    plan_id: str,
    workout_id: str,
    body: LinkActivityRequest,
    ctx_session=Depends(get_ctx_and_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    plan_result = await session.execute(
        select(TrainingPlan).where(TrainingPlan.id == plan_id, TrainingPlan.athlete_id == athlete.id)
    )
    if not plan_result.scalar_one_or_none():
        raise HTTPException(404, "Plan not found")

    workout_result = await session.execute(
        select(PlannedWorkout).where(PlannedWorkout.id == workout_id, PlannedWorkout.plan_id == plan_id)
    )
    workout = workout_result.scalar_one_or_none()
    if not workout:
        raise HTTPException(404, "Planned workout not found")

    activity_result = await session.execute(
        select(Activity).where(Activity.id == body.activity_id, Activity.athlete_id == athlete.id)
    )
    if not activity_result.scalar_one_or_none():
        raise HTTPException(404, "Activity not found")

    # Reject if this activity is already linked to a different planned workout.
    # A workout may hold many activities, but an activity belongs to only one.
    existing_link_result = await session.execute(
        select(PlannedWorkoutActivity).where(
            PlannedWorkoutActivity.activity_id == body.activity_id,
            PlannedWorkoutActivity.planned_workout_id != workout_id,
        )
    )
    if existing_link_result.scalar_one_or_none():
        raise HTTPException(409, "Activity is already linked to another planned workout")

    # Idempotent: linking the same activity to this workout again is a no-op.
    already = await session.execute(
        select(PlannedWorkoutActivity).where(
            PlannedWorkoutActivity.planned_workout_id == workout_id,
            PlannedWorkoutActivity.activity_id == body.activity_id,
        )
    )
    if not already.scalar_one_or_none():
        session.add(
            PlannedWorkoutActivity(
                planned_workout_id=workout_id, activity_id=body.activity_id
            )
        )
        await session.commit()
    await session.refresh(workout)
    # Linking is an ingest point — refresh the adherence snapshot so the score
    # moves the moment the activity lands (deterministic, not LLM-gated).
    await catch_up_adherence(athlete.id, session)
    response = PlannedWorkoutResponse.model_validate(workout)
    if workout.linked_activities:
        response.match_score = workout_match_score(workout)
    return response


@router.delete("/{plan_id}/workouts/{workout_id}/link", status_code=204)
async def unlink_workout_from_activity(
    plan_id: str,
    workout_id: str,
    activity_id: Optional[str] = None,
    ctx_session=Depends(get_ctx_and_session),
):
    """Unlink activities from a planned workout.

    With ``activity_id`` given, only that activity is unlinked; without it, every
    activity linked to the workout is unlinked.
    """
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    plan_result = await session.execute(
        select(TrainingPlan).where(TrainingPlan.id == plan_id, TrainingPlan.athlete_id == athlete.id)
    )
    if not plan_result.scalar_one_or_none():
        raise HTTPException(404, "Plan not found")

    workout_result = await session.execute(
        select(PlannedWorkout).where(PlannedWorkout.id == workout_id, PlannedWorkout.plan_id == plan_id)
    )
    workout = workout_result.scalar_one_or_none()
    if not workout:
        raise HTTPException(404, "Planned workout not found")

    conditions = [PlannedWorkoutActivity.planned_workout_id == workout_id]
    if activity_id is not None:
        conditions.append(PlannedWorkoutActivity.activity_id == activity_id)
    links_result = await session.execute(
        select(PlannedWorkoutActivity).where(*conditions)
    )
    for link in links_result.scalars().all():
        await session.delete(link)
    await session.commit()


@router.put("/{plan_id}/workouts/{workout_id}/skip", response_model=PlannedWorkoutResponse)
async def skip_workout(
    plan_id: str,
    workout_id: str,
    body: SkipWorkoutRequest,
    ctx_session=Depends(get_ctx_and_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    plan_result = await session.execute(
        select(TrainingPlan).where(TrainingPlan.id == plan_id, TrainingPlan.athlete_id == athlete.id)
    )
    if not plan_result.scalar_one_or_none():
        raise HTTPException(404, "Plan not found")

    workout_result = await session.execute(
        select(PlannedWorkout).where(PlannedWorkout.id == workout_id, PlannedWorkout.plan_id == plan_id)
    )
    workout = workout_result.scalar_one_or_none()
    if not workout:
        raise HTTPException(404, "Planned workout not found")

    if workout.is_completed:
        raise HTTPException(409, "Cannot skip a workout that has already been completed")

    workout.skip_reason = body.reason
    await session.commit()
    await session.refresh(workout)
    return PlannedWorkoutResponse.model_validate(workout)


@router.delete("/{plan_id}/workouts/{workout_id}/skip", status_code=204)
async def clear_workout_skip(
    plan_id: str,
    workout_id: str,
    ctx_session=Depends(get_ctx_and_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    plan_result = await session.execute(
        select(TrainingPlan).where(TrainingPlan.id == plan_id, TrainingPlan.athlete_id == athlete.id)
    )
    if not plan_result.scalar_one_or_none():
        raise HTTPException(404, "Plan not found")

    workout_result = await session.execute(
        select(PlannedWorkout).where(PlannedWorkout.id == workout_id, PlannedWorkout.plan_id == plan_id)
    )
    workout = workout_result.scalar_one_or_none()
    if not workout:
        raise HTTPException(404, "Planned workout not found")

    workout.skip_reason = None
    await session.commit()


@router.post("/{plan_id}/workouts", response_model=PlannedWorkoutResponse, status_code=201)
async def add_workout(
    plan_id: str,
    body: PlannedWorkoutCreate,
    ctx_session=Depends(get_ctx_and_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    plan_result = await session.execute(
        select(TrainingPlan).where(TrainingPlan.id == plan_id, TrainingPlan.athlete_id == athlete.id)
    )
    if not plan_result.scalar_one_or_none():
        raise HTTPException(404, "Plan not found")

    workout = PlannedWorkout(
        plan_id=plan_id,
        week_number=body.week_number,
        day_of_week=body.day_of_week,
        workout_type=body.workout_type,
        description=body.description,
        duration_min=body.duration_min,
        target_load=body.target_load,
    )
    session.add(workout)
    await session.commit()
    await session.refresh(workout)
    return PlannedWorkoutResponse.model_validate(workout)


@router.put("/{plan_id}/workouts/{workout_id}", response_model=PlannedWorkoutResponse)
async def update_workout(
    plan_id: str,
    workout_id: str,
    body: PlannedWorkoutUpdate,
    ctx_session=Depends(get_ctx_and_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    plan_result = await session.execute(
        select(TrainingPlan).where(TrainingPlan.id == plan_id, TrainingPlan.athlete_id == athlete.id)
    )
    if not plan_result.scalar_one_or_none():
        raise HTTPException(404, "Plan not found")

    workout_result = await session.execute(
        select(PlannedWorkout).where(PlannedWorkout.id == workout_id, PlannedWorkout.plan_id == plan_id)
    )
    workout = workout_result.scalar_one_or_none()
    if not workout:
        raise HTTPException(404, "Planned workout not found")

    if workout.is_completed:
        raise HTTPException(409, "Cannot edit a workout that has already been completed")

    for field in ("workout_type", "description", "duration_min", "target_load", "day_of_week", "week_number"):
        value = getattr(body, field)
        if value is not None:
            setattr(workout, field, value)

    await session.commit()
    await session.refresh(workout)
    return PlannedWorkoutResponse.model_validate(workout)


@router.delete("/{plan_id}/workouts/{workout_id}", status_code=204)
async def delete_workout(
    plan_id: str,
    workout_id: str,
    ctx_session=Depends(get_ctx_and_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    plan_result = await session.execute(
        select(TrainingPlan).where(TrainingPlan.id == plan_id, TrainingPlan.athlete_id == athlete.id)
    )
    if not plan_result.scalar_one_or_none():
        raise HTTPException(404, "Plan not found")

    workout_result = await session.execute(
        select(PlannedWorkout).where(PlannedWorkout.id == workout_id, PlannedWorkout.plan_id == plan_id)
    )
    workout = workout_result.scalar_one_or_none()
    if not workout:
        raise HTTPException(404, "Planned workout not found")

    if workout.is_completed:
        raise HTTPException(409, "Cannot delete a workout that has already been completed")

    await session.delete(workout)
    await session.commit()


@router.post("/{plan_id}/regenerate", response_model=TrainingPlanResponse)
async def regenerate_plan(
    plan_id: str,
    body: RegeneratePlanRequest,
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    """Replace a plan's workouts, preserving any already linked to a completed activity."""
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    result = await session.execute(
        select(TrainingPlan)
        .where(TrainingPlan.id == plan_id, TrainingPlan.athlete_id == athlete.id)
        .options(selectinload(TrainingPlan.workouts))
    )
    plan = result.scalar_one_or_none()
    if not plan:
        raise HTTPException(404, "Plan not found")

    config = clamp_plan_params(body.config) if body.config else None
    num_weeks = body.weeks if body.weeks is not None else (plan.weeks or 8)
    goal = body.goal if body.goal is not None else plan.goal

    # Preserve completed workouts; drop the rest. Reassigning the delete-orphan
    # collection schedules the removed (non-completed) rows for deletion — calling
    # session.delete() while they remain in the loaded collection does not stick.
    preserved = [pw for pw in plan.workouts if pw.is_completed]
    occupied = {(pw.week_number, pw.day_of_week) for pw in preserved}
    plan.workouts = list(preserved)
    await session.flush()

    def _add(week_num: int, day_of_week: int, **fields) -> None:
        if (week_num, day_of_week) in occupied:
            return
        plan.workouts.append(PlannedWorkout(
            week_number=week_num, day_of_week=day_of_week, **fields,
        ))

    # When regenerating via the LLM (pre-built or server-side), keep the actual
    # generated weeks so week_meta is summarised from them, not a rule-based rebuild.
    llm_weeks_data: list[list[dict]] | None = None

    if body.llm_weeks:
        plan.generation_method = "llm"
        llm_weeks_data = _llm_weeks_to_day_dicts(body.llm_weeks)
        for week_num, week_days in enumerate(body.llm_weeks, start=1):
            for day in week_days:
                _add(
                    week_num, day.day_of_week,
                    workout_type=day.workout_type,
                    description=day.description,
                    duration_min=day.duration_min,
                    target_load=day.target_load,
                )
    elif body.use_llm:
        if not config:
            raise HTTPException(400, "A plan config (training days and types) is required for LLM generation")
        instance_result = await registry_session.execute(select(InstanceSettings).limit(1))
        instance = instance_result.scalar_one_or_none()
        access = await check_llm_access(ctx, athlete, instance, registry_session)
        if not access.allowed:
            raise subscription_required_error()
        try:
            weeks_data = await generate_plan_weeks_llm(
                athlete=athlete,
                config=config,
                num_weeks=num_weeks,
                goal=goal,
                session=session,
                instance=instance,
                user_id=ctx.user_id,
                allow_instance_fallback=(access.mode != "byok"),
            )
        except LlmConfigError as exc:
            raise HTTPException(LLM_ERROR_STATUS.get(exc.code, 400), str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(503, f"LLM plan generation failed: {exc}") from exc
        plan.generation_method = "llm"
        llm_weeks_data = weeks_data
        for week_num, week_days in enumerate(weeks_data, start=1):
            for day in week_days:
                _add(week_num, **day)
    else:
        plan.generation_method = "rule_based"
        for pw in build_workout_rows(plan.id, num_weeks, goal, config):
            if (pw.week_number, pw.day_of_week) in occupied:
                continue
            plan.workouts.append(pw)

    plan.weeks = num_weeks
    plan.goal = goal
    if config is not None:
        plan.config = config.model_dump()
        # Summarise LLM regenerations from the actual generated weeks; only the
        # rule-based path derives week_meta from a fresh rule-based build.
        if llm_weeks_data is not None:
            plan.week_meta = week_meta_from_weeks(config, llm_weeks_data)
        else:
            plan.week_meta = week_meta_for(config, num_weeks)
    if plan.start_date:
        plan.end_date = plan.start_date + timedelta(weeks=num_weeks) - timedelta(days=1)

    await session.commit()

    result = await session.execute(
        select(TrainingPlan)
        .where(TrainingPlan.id == plan.id)
        .options(selectinload(TrainingPlan.workouts))
    )
    plan = result.scalar_one()
    return _plan_response_with_adherence(plan)


@router.post("/{plan_id}/generate-upcoming/workouts", response_model=GenerateUpcomingWorkoutsResponse)
async def generate_upcoming_workouts(
    plan_id: str,
    body: GenerateUpcomingWorkoutsRequest,
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    """Synthesize structured workouts for the plan's upcoming days (no upload).

    For each planned workout falling within the upcoming-week window (today→+6;
    rest days and unstructured rows excluded), ensure a structured
    ``WorkoutDefinition`` exists — generating one via the LLM and caching it on
    ``PlannedWorkout.workout_definition_id`` when missing. The generated workouts
    show up in the Workouts tab, where they can be reviewed, edited, and uploaded
    individually. Returns a per-workout summary.
    """
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    result = await session.execute(
        select(TrainingPlan)
        .where(TrainingPlan.id == plan_id, TrainingPlan.athlete_id == athlete.id)
        .options(selectinload(TrainingPlan.workouts))
    )
    plan = result.scalar_one_or_none()
    if not plan:
        raise HTTPException(404, "Plan not found")
    if not plan.start_date:
        raise HTTPException(400, "Plan has no start date")

    # Compute the [start, end] selection window, clamped to today→+6 days.
    today = datetime.now(timezone.utc).date()
    window_start = today
    window_end = today + timedelta(days=_GENERATE_WINDOW_DAYS)
    if body.start and body.start > window_start:
        window_start = body.start
    if body.end and body.end < window_end:
        window_end = body.end

    instance_result = await registry_session.execute(select(InstanceSettings).limit(1))
    instance = instance_result.scalar_one_or_none()
    access = await check_llm_access(ctx, athlete, instance, registry_session)
    if not access.allowed:
        raise subscription_required_error()

    # Select in-window planned workouts, ordered by date.
    selected: list[tuple[PlannedWorkout, date]] = []
    for pw in plan.workouts:
        pdate = _planned_date(plan.start_date, pw.week_number, pw.day_of_week)
        if window_start <= pdate <= window_end:
            selected.append((pw, pdate))
    selected.sort(key=lambda item: item[1])

    results: list[GenerateUpcomingResultItem] = []

    for pw, pdate in selected:
        wtype = (pw.workout_type or "").lower()
        if wtype in ("", "rest") or (pw.duration_min is None and pw.target_load is None):
            results.append(GenerateUpcomingResultItem(
                planned_workout_id=pw.id, date=pdate, workout_type=pw.workout_type,
                status="skipped", reason="rest_or_unstructured",
            ))
            continue

        # Reuse a cached definition unless missing or a refresh was requested.
        existing: WorkoutDefinition | None = None
        if pw.workout_definition_id and not body.refresh:
            wd_result = await session.execute(
                select(WorkoutDefinition).where(
                    WorkoutDefinition.id == pw.workout_definition_id,
                    WorkoutDefinition.athlete_id == athlete.id,
                )
            )
            existing = wd_result.scalar_one_or_none()

        if existing is not None:
            results.append(GenerateUpcomingResultItem(
                planned_workout_id=pw.id, date=pdate, workout_type=pw.workout_type,
                workout_definition_id=existing.id,
                status="skipped", reason="already_generated",
            ))
            continue

        try:
            workout = await generate_workout_definition_llm(
                athlete=athlete,
                planned_workout=pw,
                session=session,
                instance=instance,
                user_id=ctx.user_id,
                allow_instance_fallback=(access.mode != "byok"),
            )
        except LlmConfigError as exc:
            raise HTTPException(LLM_ERROR_STATUS.get(exc.code, 400), str(exc)) from exc
        except ValueError as exc:
            # LLM not configured — no workout can be generated; fail clearly.
            raise HTTPException(400, str(exc)) from exc
        except WorkoutGenerationError as exc:
            results.append(GenerateUpcomingResultItem(
                planned_workout_id=pw.id, date=pdate, workout_type=pw.workout_type,
                status="failed", reason=f"generation_failed: {exc}",
            ))
            continue
        except httpx.HTTPError:
            # Transient LLM connectivity/HTTP error — skip this day, keep the batch going.
            results.append(GenerateUpcomingResultItem(
                planned_workout_id=pw.id, date=pdate, workout_type=pw.workout_type,
                status="failed", reason="generation_failed: llm_unavailable",
            ))
            continue

        results.append(GenerateUpcomingResultItem(
            planned_workout_id=pw.id, date=pdate, workout_type=pw.workout_type,
            workout_definition_id=workout.id, status="generated",
        ))

    await session.commit()
    return GenerateUpcomingWorkoutsResponse(results=results)


@router.delete("/{plan_id}", status_code=204)
async def delete_plan(
    plan_id: str,
    ctx_session=Depends(get_ctx_and_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    result = await session.execute(
        select(TrainingPlan)
        .where(TrainingPlan.id == plan_id, TrainingPlan.athlete_id == athlete.id)
    )
    plan = result.scalar_one_or_none()
    if not plan:
        raise HTTPException(404, "Plan not found")
    await session.delete(plan)
    await session.commit()
