"""Achievements & streaks API (issue #33).

Deterministic and always-on — not gated behind the LLM subscription, the same
stance as the plan-adherence scores. Each read runs a cheap catch-up first (as
``GET /plans/{id}/adherence`` does) so the badges are fresh even for a user who
hasn't loaded the dashboard yet.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException

from backend.app.core.deps import get_ctx_and_session
from backend.app.models.user_orm import AchievementUnlock, Athlete
from backend.app.schemas.achievements import (
    AchievementDefinition,
    AchievementsResponse,
    AchievementUnlockResponse,
    StreakResponse,
)
from backend.app.services.achievements import (
    compute_achievements,
    gamification_enabled,
    recompute_achievements,
)
from openkoutsi.achievements import CATALOGUE

router = APIRouter(prefix="/achievements", tags=["achievements"])


async def _get_athlete(global_user_id: str, session: AsyncSession) -> Athlete:
    result = await session.execute(
        select(Athlete).where(Athlete.global_user_id == global_user_id)
    )
    athlete = result.scalar_one_or_none()
    if athlete is None:
        raise HTTPException(status_code=404, detail="Athlete profile not found")
    return athlete


@router.get("", response_model=AchievementsResponse,
            operation_id="getAchievements", summary="Achievements, progress and streaks")
async def get_achievements(ctx_session=Depends(get_ctx_and_session)):
    """Catalogue, earned tiers, progress toward the locked ones, and streaks.

    Achievements whose data requirement the athlete cannot meet (elevation
    without a barometric FIT, Load without power or HR) are left out of the
    catalogue entirely rather than shown permanently locked.
    """
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    if not gamification_enabled(athlete):
        return AchievementsResponse(
            catalogue=[], unlocked=[], progress={}, streaks=[], disabled=True
        )

    await recompute_achievements(athlete.id, session)
    comp = await compute_achievements(athlete, session)

    catalogue = [
        AchievementDefinition(
            id=d.id, category=d.category, tiers=list(d.tiers),
            unit=d.unit, requires=d.requires,
        )
        for d in CATALOGUE
        if comp.is_available(d.id)
    ]
    available_ids = {d.id for d in catalogue}

    rows = (
        await session.execute(
            select(AchievementUnlock)
            .where(AchievementUnlock.athlete_id == athlete.id)
            .order_by(AchievementUnlock.achieved_on)
        )
    ).scalars().all()

    return AchievementsResponse(
        catalogue=catalogue,
        unlocked=[AchievementUnlockResponse.model_validate(r) for r in rows],
        progress={k: v for k, v in comp.progress.items() if k in available_ids},
        streaks=[
            StreakResponse(
                id=key, current=state.current, longest=state.longest,
                in_progress=state.in_progress,
            )
            for key, state in comp.streaks.items()
            if key in available_ids
        ],
    )


@router.get("/streaks", response_model=list[StreakResponse],
            operation_id="getStreaks", summary="Current and longest streaks")
async def get_streaks(ctx_session=Depends(get_ctx_and_session)):
    """Just the streaks, for the dashboard card."""
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    if not gamification_enabled(athlete):
        return []

    comp = await compute_achievements(athlete, session)
    return [
        StreakResponse(
            id=key, current=state.current, longest=state.longest,
            in_progress=state.in_progress,
        )
        for key, state in comp.streaks.items()
        if comp.is_available(key)
    ]


@router.post("/seen", status_code=204,
             operation_id="markAchievementsSeen", summary="Clear the new-achievement marker")
async def mark_seen(ctx_session=Depends(get_ctx_and_session)):
    """Mark every unlock as seen, so the UI stops flagging them as new.

    Deliberately separate from ``notified``: dismissing the marker here must not
    cancel an inbox message the server hasn't sent yet.
    """
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    rows = (
        await session.execute(
            select(AchievementUnlock).where(
                AchievementUnlock.athlete_id == athlete.id,
                AchievementUnlock.seen.is_(False),
            )
        )
    ).scalars().all()
    for row in rows:
        row.seen = True
    if rows:
        await session.commit()
