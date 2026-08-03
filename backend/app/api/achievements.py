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

from backend.app.core.deps import get_ctx_session_athlete
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


@router.get("", response_model=AchievementsResponse,
            operation_id="getAchievements", summary="Achievements, progress and streaks")
async def get_achievements(ctx_athlete=Depends(get_ctx_session_athlete)):
    """Catalogue, earned tiers, progress toward the locked ones, and streaks.

    Achievements whose data requirement the athlete cannot meet (elevation
    without a barometric FIT, Load without power or HR) are left out of the
    catalogue entirely rather than shown permanently locked.
    """
    ctx, session, athlete = ctx_athlete

    if not gamification_enabled(athlete):
        return AchievementsResponse(
            catalogue=[], unlocked=[], progress={}, streaks=[], disabled=True
        )

    # One pass: the reconcile hands back the computation it already built, so a
    # read scans the athlete's history once rather than twice.
    _, comp = await recompute_achievements(athlete.id, session, athlete=athlete)
    if comp is None:
        return AchievementsResponse(
            catalogue=[], unlocked=[], progress={}, streaks=[], disabled=True
        )

    catalogue = [
        AchievementDefinition(
            id=d.id, category=d.category, tiers=list(d.tiers),
            unit=d.unit, requires=d.requires,
            threshold=d.threshold, threshold_unit=d.threshold_unit,
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
        # Filtered like its three siblings below, so the response is internally
        # consistent by construction: every `unlocked` id has a `catalogue` entry
        # to render it with. The reconcile deliberately persists unlocks without
        # regard to availability, so nothing else guarantees that. The row stays
        # in the DB (and in the data export) — only the rendering is gated, which
        # is what availability has always meant.
        unlocked=[
            AchievementUnlockResponse.model_validate(r)
            for r in rows
            if r.achievement_id in available_ids
        ],
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
async def get_streaks(ctx_athlete=Depends(get_ctx_session_athlete)):
    """Just the streaks, for the dashboard card."""
    ctx, session, athlete = ctx_athlete

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


# Deliberately no staleness gate on the reads above, unlike
# ``get_training_status``. Some achievements turn on time passing rather than on
# an upload — a plan becomes "completed" the day after its end date, and a streak
# lapses when a week closes — so a read that skipped the recompute would show
# stale badges to an athlete who simply hasn't ridden lately. Now that a read
# costs one pass rather than two, paying it is the cheaper trade.


@router.post("/seen", status_code=204,
             operation_id="markAchievementsSeen", summary="Clear the new-achievement marker")
async def mark_seen(ctx_athlete=Depends(get_ctx_session_athlete)):
    """Mark every unlock as seen, so the UI stops flagging them as new.

    Purely a UI marker — it has no bearing on the inbox message, which is emitted
    once by the reconcile that first inserted the row.
    """
    ctx, session, athlete = ctx_athlete

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
