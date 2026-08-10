import asyncio
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.deps import get_ctx_session_athlete
from backend.app.core.scopes import pat_forbidden, pat_scopes
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
from backend.app.services.achievements import recompute_achievements_safe
from backend.app.services.stranded_runs import pending_timed_out, settle_goal_guidance


router = APIRouter(
    prefix="/goals",
    tags=["goals"],
    dependencies=[pat_scopes(read="goals:read", write="goals:write")],
)


@router.get("", response_model=Page[GoalResponse],
            operation_id="listGoals", summary="List goals")
async def list_goals(
    ctx_athlete=Depends(get_ctx_session_athlete),
    params: PageParams = Depends(paginate_params),
):
    ctx, session, athlete = ctx_athlete
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
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete
    goal = Goal(id=str(uuid.uuid4()), athlete_id=athlete.id, **body.model_dump())
    session.add(goal)
    await session.commit()
    await session.refresh(goal)
    return goal


@router.put("/{goal_id}", response_model=GoalResponse)
async def update_goal(
    goal_id: str,
    body: GoalUpdate,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete
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
    # Marking a goal achieved can unlock a badge (issue #33); un-marking it
    # takes the badge back, since unlocks track the data rather than events.
    await recompute_achievements_safe(athlete.id, session)
    return goal


@router.delete("/{goal_id}", status_code=204)
async def delete_goal(
    goal_id: str,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete
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
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete
    goal = await _get_owned_goal(goal_id, athlete, session)

    # Recover from a stuck "pending" state: if the run hasn't shown progress
    # within the timeout window, reset to "error" so the user can retry. The
    # window is an inactivity budget — the generator touches the timestamp on
    # every progress commit (issue #91) — so a slow but healthy stream is no
    # longer declared dead while it is still writing. A NULL updated_at with
    # status "pending" is treated as immediately timed out.
    if goal.guidance_status == "pending":
        now_utc = datetime.now(timezone.utc)
        if pending_timed_out(goal.guidance_updated_at, now_utc):
            settle_goal_guidance(goal, now_utc)
            await session.commit()

    return GoalGuidanceResponse(
        status=goal.guidance_status,
        verdict=goal.guidance_verdict,
        guidance=goal.guidance,
        updated_at=goal.guidance_updated_at,
    )


@router.post("/{goal_id}/guidance", status_code=202, dependencies=[pat_forbidden()],
             operation_id="triggerGoalGuidance", summary="Trigger AI guidance for a goal")
async def trigger_goal_guidance(
    goal_id: str,
    body: GoalGuidanceBody = GoalGuidanceBody(),
    ctx_athlete=Depends(get_ctx_session_athlete),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    ctx, session, athlete = ctx_athlete
    goal = await _get_owned_goal(goal_id, athlete, session)

    # Issue #9 gate (goal guidance is always instance-paid).
    from backend.app.services.llm_access import check_llm_access, subscription_required_error
    instance = (
        await registry_session.execute(select(InstanceSettings).limit(1))
    ).scalar_one_or_none()
    access = await check_llm_access(ctx, athlete, instance, registry_session)
    if not access.allowed:
        raise subscription_required_error(access)

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
