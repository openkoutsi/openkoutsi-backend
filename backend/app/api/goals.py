import asyncio
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.deps import get_ctx_and_session
from backend.app.db.registry import get_registry_session
from backend.app.models.registry_orm import InstanceSettings
from backend.app.models.user_orm import Athlete, Goal
from backend.app.schemas.goals import (
    GoalCreate,
    GoalGuidanceBody,
    GoalGuidanceResponse,
    GoalResponse,
    GoalUpdate,
)
from backend.app.schemas.pagination import Page, PageParams, paginate_params

router = APIRouter(prefix="/goals", tags=["goals"])

# Recover from a stuck "pending" guidance state: if the background task hasn't
# completed within this window, reset to "error" so the user can retry.
_PENDING_TIMEOUT_MINUTES = 30


async def _get_athlete(global_user_id: str, session: AsyncSession) -> Athlete:
    result = await session.execute(select(Athlete).where(Athlete.global_user_id == global_user_id))
    athlete = result.scalar_one_or_none()
    if athlete is None:
        raise HTTPException(status_code=404, detail="Athlete profile not found")
    return athlete


@router.get("", response_model=Page[GoalResponse],
            operation_id="listGoals", summary="List goals")
async def list_goals(
    ctx_session=Depends(get_ctx_and_session),
    params: PageParams = Depends(paginate_params),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    total = (await session.execute(
        select(func.count()).select_from(Goal).where(Goal.athlete_id == athlete.id)
    )).scalar_one()
    result = await session.execute(
        select(Goal)
        .where(Goal.athlete_id == athlete.id)
        .order_by(Goal.created_at.desc())
        .offset(params.offset)
        .limit(params.page_size)
    )
    items = [GoalResponse.model_validate(g) for g in result.scalars().all()]
    return Page.build(items, total, params.page, params.page_size)


@router.post("", response_model=GoalResponse, status_code=201)
async def create_goal(
    body: GoalCreate,
    ctx_session=Depends(get_ctx_and_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    goal = Goal(id=str(uuid.uuid4()), athlete_id=athlete.id, **body.model_dump())
    session.add(goal)
    await session.commit()
    await session.refresh(goal)
    return goal


@router.put("/{goal_id}", response_model=GoalResponse)
async def update_goal(
    goal_id: str,
    body: GoalUpdate,
    ctx_session=Depends(get_ctx_and_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    result = await session.execute(
        select(Goal).where(Goal.id == goal_id, Goal.athlete_id == athlete.id)
    )
    goal = result.scalar_one_or_none()
    if goal is None:
        raise HTTPException(status_code=404, detail="Goal not found")

    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(goal, field, value)

    await session.commit()
    await session.refresh(goal)
    return goal


@router.delete("/{goal_id}", status_code=204)
async def delete_goal(
    goal_id: str,
    ctx_session=Depends(get_ctx_and_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    result = await session.execute(
        select(Goal).where(Goal.id == goal_id, Goal.athlete_id == athlete.id)
    )
    goal = result.scalar_one_or_none()
    if goal is None:
        raise HTTPException(status_code=404, detail="Goal not found")

    await session.delete(goal)
    await session.commit()


async def _get_owned_goal(goal_id: str, athlete: Athlete, session: AsyncSession) -> Goal:
    result = await session.execute(
        select(Goal).where(Goal.id == goal_id, Goal.athlete_id == athlete.id)
    )
    goal = result.scalar_one_or_none()
    if goal is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    return goal


@router.get("/{goal_id}/guidance", response_model=GoalGuidanceResponse,
            operation_id="getGoalGuidance", summary="Get AI guidance for a goal")
async def get_goal_guidance(
    goal_id: str,
    ctx_session=Depends(get_ctx_and_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    goal = await _get_owned_goal(goal_id, athlete, session)

    # Recover from a stuck "pending" state: if the task hasn't completed within
    # the timeout window, reset to "error" so the user can retry. A NULL
    # updated_at with status "pending" is treated as immediately timed out.
    if goal.guidance_status == "pending":
        now_utc = datetime.now(timezone.utc)
        updated_at = goal.guidance_updated_at
        if updated_at is not None:
            aware = updated_at if updated_at.tzinfo else updated_at.replace(tzinfo=timezone.utc)
            timed_out = (now_utc - aware.astimezone(timezone.utc)).total_seconds() > _PENDING_TIMEOUT_MINUTES * 60
        else:
            timed_out = True
        if timed_out:
            goal.guidance_status = "error"
            goal.guidance_updated_at = now_utc
            await session.commit()

    return GoalGuidanceResponse(
        status=goal.guidance_status,
        verdict=goal.guidance_verdict,
        guidance=goal.guidance,
        updated_at=goal.guidance_updated_at,
    )


@router.post("/{goal_id}/guidance", status_code=202,
             operation_id="triggerGoalGuidance", summary="Trigger AI guidance for a goal")
async def trigger_goal_guidance(
    goal_id: str,
    body: GoalGuidanceBody = GoalGuidanceBody(),
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    goal = await _get_owned_goal(goal_id, athlete, session)

    # Issue #9 gate (goal guidance is always instance-paid).
    from backend.app.services.llm_access import check_llm_access, subscription_required_error
    instance = (
        await registry_session.execute(select(InstanceSettings).limit(1))
    ).scalar_one_or_none()
    access = await check_llm_access(ctx, athlete, instance, registry_session)
    if not access.allowed:
        raise subscription_required_error()

    if goal.guidance_status == "pending":
        return {"status": "pending"}

    goal.guidance_status = "pending"
    goal.guidance = None
    goal.guidance_verdict = None
    goal.guidance_updated_at = datetime.now(timezone.utc)
    await session.commit()

    from backend.app.services.llm_goal_guidance import generate_goal_guidance_bg
    asyncio.create_task(
        generate_goal_guidance_bg(athlete.id, goal.id, ctx.user_id, body.locale)
    )
    return {"status": "pending"}
