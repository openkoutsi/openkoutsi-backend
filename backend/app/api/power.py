from datetime import date, datetime, timedelta, timezone
from itertools import groupby
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.deps import get_ctx_and_session
from backend.app.models.team_orm import Activity, ActivityPowerBest, Athlete, WeightLog
from backend.app.schemas.power import AllTimePowerBestsResponse, FtpEstimateResponse, PowerBestEntry
from openkoutsi.training_math import (
    CP_FIT_DURATIONS,
    POWER_BEST_DURATIONS,
    estimate_cp_wprime,
    estimate_ftp_simple,
)

router = APIRouter(prefix="/power", tags=["power"])

TOP_N = 3


async def _get_athlete(global_user_id: str, session: AsyncSession) -> Athlete:
    result = await session.execute(select(Athlete).where(Athlete.global_user_id == global_user_id))
    athlete = result.scalar_one_or_none()
    if athlete is None:
        raise HTTPException(status_code=404, detail="Athlete profile not found")
    return athlete


@router.get("/bests", response_model=AllTimePowerBestsResponse)
async def get_power_bests(
    days: Optional[int] = Query(None, ge=1, description="Restrict to bests from the past N days. Omit for all-time."),
    ctx_session=Depends(get_ctx_and_session),
):
    """
    Return the top-3 best average power for each standard duration,
    ordered by (duration_s asc, rank asc).  Durations with no data are omitted.
    Pass ?days=90/180/365 to restrict to a rolling window; omit for all-time.
    """
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    # Load weight log (sorted ascending by date for the lookup below)
    wl_rows = await session.execute(
        select(WeightLog)
        .where(WeightLog.athlete_id == athlete.id)
        .order_by(WeightLog.effective_date)
    )
    weight_log: list[tuple[date, float]] = [
        (w.effective_date, w.weight_kg) for w in wl_rows.scalars().all()
    ]

    def _effective_weight(activity_date: Optional[date]) -> Optional[float]:
        """Return the most recent weight whose effective_date <= activity_date."""
        if not activity_date or not weight_log:
            return None
        result: Optional[float] = None
        for eff_date, w_kg in weight_log:
            if eff_date <= activity_date:
                result = w_kg
            else:
                break
        return result

    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=days)
        if days is not None
        else None
    )

    where_clauses = [ActivityPowerBest.athlete_id == athlete.id]
    if cutoff is not None:
        where_clauses.append(ActivityPowerBest.activity_start_time >= cutoff)

    rows = await session.execute(
        select(ActivityPowerBest, Activity.name)
        .join(Activity, Activity.id == ActivityPowerBest.activity_id)
        .where(*where_clauses)
        .order_by(ActivityPowerBest.duration_s, ActivityPowerBest.power_w.desc())
    )
    records = rows.all()

    # Group by duration_s in the order they come from the query (already sorted)
    entries: list[PowerBestEntry] = []
    for _, group in groupby(records, key=lambda r: r[0].duration_s):
        for rank, (best, activity_name) in enumerate(group, start=1):
            if rank > TOP_N:
                break
            act_date = best.activity_start_time.date() if best.activity_start_time else None
            entries.append(
                PowerBestEntry(
                    duration_s=best.duration_s,
                    rank=rank,
                    power_w=round(best.power_w, 1),
                    activity_id=best.activity_id,
                    activity_name=activity_name,
                    activity_start_time=best.activity_start_time,
                    weight_kg=_effective_weight(act_date),
                )
            )

    # Preserve canonical duration order (POWER_BEST_DURATIONS) rather than
    # whatever order the DB happened to return.
    duration_order = {d: i for i, d in enumerate(POWER_BEST_DURATIONS)}
    entries.sort(key=lambda e: (duration_order.get(e.duration_s, 9999), e.rank))

    return AllTimePowerBestsResponse(bests=entries)


@router.get("/ftp-estimate", response_model=FtpEstimateResponse)
async def get_ftp_estimate(
    days: Optional[int] = Query(None, ge=1, description="Estimate from bests in the past N days. Omit for all-time."),
    ctx_session=Depends(get_ctx_and_session),
):
    """
    Estimate FTP from the athlete's power curve using two methods:

    - Simple: 95% of the 20-minute (1200s) best power.
    - Critical Power: linear work–time fit over the 2–20 minute bests (CP).

    Both estimates use the rank-1 (single best) power per duration.  Pass
    ?days=90/180/365 to estimate from a rolling window; omit for all-time.
    """
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=days)
        if days is not None
        else None
    )

    where_clauses = [
        ActivityPowerBest.athlete_id == athlete.id,
        ActivityPowerBest.duration_s.in_(CP_FIT_DURATIONS),
    ]
    if cutoff is not None:
        where_clauses.append(ActivityPowerBest.activity_start_time >= cutoff)

    rows = await session.execute(
        select(ActivityPowerBest.duration_s, ActivityPowerBest.power_w)
        .where(*where_clauses)
        .order_by(ActivityPowerBest.duration_s, ActivityPowerBest.power_w.desc())
    )

    # Keep the single best (rank-1) power per duration.
    rank1: dict[int, float] = {}
    for duration_s, power_w in rows.all():
        if duration_s not in rank1:
            rank1[duration_s] = power_w

    twenty_min_power = rank1.get(1200)
    ftp_simple_raw = estimate_ftp_simple(twenty_min_power)
    cp, w_prime = estimate_cp_wprime(rank1)

    return FtpEstimateResponse(
        twenty_min_power=round(twenty_min_power, 1) if twenty_min_power is not None else None,
        ftp_simple=round(ftp_simple_raw) if ftp_simple_raw is not None else None,
        simple_available=ftp_simple_raw is not None,
        cp=round(cp, 1) if cp is not None else None,
        w_prime=round(w_prime) if w_prime is not None else None,
        ftp_cp=round(cp) if cp is not None else None,
        cp_available=cp is not None,
    )
