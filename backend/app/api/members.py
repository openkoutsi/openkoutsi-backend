"""
Coach-access endpoints: allow coaches and administrators to view any
team member's activity list and athlete profile.

These endpoints use the combined get_ctx_and_session dependency (team DB)
and the get_registry_session dependency (registry DB) to look up the
member's global_user_id from the registry, then query the team DB.
"""
from typing import Optional
from datetime import date, datetime, time

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import TeamContext, get_current_user
from backend.app.core.deps import get_ctx_and_session
from backend.app.db.registry import get_registry_session
from backend.app.models.registry_orm import TeamMembership
from backend.app.models.team_orm import Activity, ActivitySource, Athlete
from backend.app.schemas.activities import ActivityListResponse, ActivityResponse
from backend.app.schemas.athlete import AthleteResponse
from backend.app.api.athlete import _athlete_response

router = APIRouter(prefix="/members", tags=["members"])


def _require_coach_or_admin(ctx: TeamContext) -> None:
    if not (ctx.is_coach or ctx.is_admin):
        raise HTTPException(status_code=403, detail="Coach or administrator role required")


async def _resolve_member_athlete(
    member_user_id: str,
    ctx: TeamContext,
    team_session: AsyncSession,
    registry_session: AsyncSession,
) -> Athlete:
    """Verify `member_user_id` is a member of the caller's team and return their Athlete row."""
    mb_result = await registry_session.execute(
        select(TeamMembership).where(
            TeamMembership.team_id == ctx.team_id,
            TeamMembership.user_id == member_user_id,
        )
    )
    if mb_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Member not found")

    athlete_result = await team_session.execute(
        select(Athlete).where(Athlete.global_user_id == member_user_id)
    )
    athlete = athlete_result.scalar_one_or_none()
    if athlete is None:
        raise HTTPException(status_code=404, detail="Athlete profile not found")
    return athlete


@router.get("/{member_user_id}/athlete", response_model=AthleteResponse)
async def get_member_athlete(
    member_user_id: str,
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    """Return the athlete profile of any team member (coach/admin only)."""
    ctx, session = ctx_session
    _require_coach_or_admin(ctx)
    athlete = await _resolve_member_athlete(member_user_id, ctx, session, registry_session)
    return _athlete_response(athlete, [], ctx.team_id)


@router.get("/{member_user_id}/activities", response_model=ActivityListResponse)
async def list_member_activities(
    member_user_id: str,
    start: Optional[date] = Query(None),
    end: Optional[date] = Query(None),
    sport_type: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    """List activities for any team member (coach/admin only)."""
    ctx, session = ctx_session
    _require_coach_or_admin(ctx)
    athlete = await _resolve_member_athlete(member_user_id, ctx, session, registry_session)

    base_query = select(Activity).where(Activity.athlete_id == athlete.id)
    if start:
        base_query = base_query.where(Activity.start_time >= datetime.combine(start, time.min))
    if end:
        base_query = base_query.where(Activity.start_time <= datetime.combine(end, time.max))
    if sport_type:
        base_query = base_query.where(Activity.sport_type == sport_type)

    count_result = await session.execute(
        select(func.count()).select_from(base_query.subquery())
    )
    total = count_result.scalar_one()

    items_result = await session.execute(
        base_query.order_by(Activity.start_time.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    items = [ActivityResponse.model_validate(a) for a in items_result.scalars().all()]
    return ActivityListResponse(items=items, total=total, page=page, page_size=page_size)
