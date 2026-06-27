import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.deps import get_ctx_and_session
from backend.app.models.user_orm import Athlete, Goal
from backend.app.schemas.goals import GoalCreate, GoalResponse, GoalUpdate
from backend.app.schemas.pagination import Page, PageParams, paginate_params

router = APIRouter(prefix="/goals", tags=["goals"])


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
