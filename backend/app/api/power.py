from datetime import datetime, timedelta, timezone
from itertools import groupby
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.deps import get_ctx_and_session
from backend.app.models.user_orm import Activity, ActivityPowerBest, Athlete
from backend.app.schemas.power import AllTimePowerBestsResponse, FtpEstimateResponse, PowerBestEntry
from backend.app.services.weight import effective_weight_for, load_weight_log, w_per_kg
from openkoutsi.training_math import (
    CP_FIT_DURATIONS,
    POWER_BEST_DURATIONS,
    estimate_cp_wprime,
    estimate_ftp_simple,
)

router = APIRouter(prefix="/metrics", tags=["metrics"])

TOP_N = 3

Metric = Literal["watts", "wkg"]


async def _get_athlete(global_user_id: str, session: AsyncSession) -> Athlete:
    result = await session.execute(select(Athlete).where(Athlete.global_user_id == global_user_id))
    athlete = result.scalar_one_or_none()
    if athlete is None:
        raise HTTPException(status_code=404, detail="Athlete profile not found")
    return athlete


async def all_time_power_bests(
    athlete: Athlete,
    session: AsyncSession,
    days: Optional[int] = None,
    metric: Metric = "watts",
) -> list[PowerBestEntry]:
    """Top-3 best efforts per standard duration for an athlete.

    ``metric="watts"`` ranks each duration by absolute power; ``metric="wkg"``
    ranks by watts-per-kg using the effective bodyweight at the time of each
    effort, and omits efforts with no known weight. Ordered by (duration_s asc,
    rank asc); durations with no qualifying data are omitted. Pass ``days`` to
    restrict to a rolling window; omit for all-time. Shared by the
    ``/bests/power`` route and the data export.
    """
    # Effective weight is recomputed from the log on read, so the curve reflects
    # the current weight history even if the stored per-row values are stale.
    weight_log = await load_weight_log(athlete.id, session)

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
        .order_by(ActivityPowerBest.duration_s)
    )
    records = rows.all()

    def _weight(best: ActivityPowerBest) -> Optional[float]:
        act_date = best.activity_start_time.date() if best.activity_start_time else None
        return effective_weight_for(weight_log, act_date)

    entries: list[PowerBestEntry] = []
    for _, group in groupby(records, key=lambda r: r[0].duration_s):
        candidates = []
        for best, activity_name in group:
            weight = _weight(best)
            wkg = w_per_kg(best.power_w, weight)
            if metric == "wkg" and wkg is None:
                # No contemporaneous weight — can't rank this effort by W/kg.
                continue
            candidates.append((best, activity_name, weight, wkg))

        # Rank within the duration by the chosen metric, ties by earlier effort.
        sort_key = (
            (lambda c: (-(c[3] or 0.0), c[0].activity_start_time or datetime.max.replace(tzinfo=timezone.utc)))
            if metric == "wkg"
            else (lambda c: (-c[0].power_w, c[0].activity_start_time or datetime.max.replace(tzinfo=timezone.utc)))
        )
        candidates.sort(key=sort_key)

        for rank, (best, activity_name, weight, wkg) in enumerate(candidates[:TOP_N], start=1):
            entries.append(
                PowerBestEntry(
                    duration_s=best.duration_s,
                    rank=rank,
                    power_w=round(best.power_w, 1),
                    activity_id=best.activity_id,
                    activity_name=activity_name,
                    activity_start_time=best.activity_start_time,
                    weight_kg=weight,
                    w_per_kg=round(wkg, 3) if wkg is not None else None,
                )
            )

    # Preserve canonical duration order (POWER_BEST_DURATIONS) rather than
    # whatever order the DB happened to return.
    duration_order = {d: i for i, d in enumerate(POWER_BEST_DURATIONS)}
    entries.sort(key=lambda e: (duration_order.get(e.duration_s, 9999), e.rank))

    return entries


@router.get("/bests/power", response_model=AllTimePowerBestsResponse,
            operation_id="getPowerBests", summary="All-time power bests")
async def get_power_bests(
    days: Optional[int] = Query(None, ge=1, description="Restrict to bests from the past N days. Omit for all-time."),
    metric: Metric = Query("watts", description="Rank by absolute 'watts' or by 'wkg' (watts per kg at the time of each effort)."),
    ctx_session=Depends(get_ctx_and_session),
):
    """
    Return the top-3 best efforts for each standard duration,
    ordered by (duration_s asc, rank asc).  Durations with no data are omitted.

    Pass ?metric=wkg to rank by watts-per-kg using the effective bodyweight at
    the time of each effort (efforts with no known weight are omitted); the
    default ?metric=watts ranks by absolute power.  Pass ?days=90/180/365 to
    restrict to a rolling window; omit for all-time.
    """
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    entries = await all_time_power_bests(athlete, session, days=days, metric=metric)
    return AllTimePowerBestsResponse(bests=entries)


@router.get("/ftp", response_model=FtpEstimateResponse,
            operation_id="getFtpEstimate", summary="Current FTP estimate")
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
