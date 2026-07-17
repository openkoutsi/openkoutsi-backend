from datetime import datetime, timedelta, timezone
from itertools import groupby
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.deps import get_ctx_and_session
from backend.app.models.user_orm import Activity, ActivityPowerBest, Athlete
from backend.app.schemas.power import (
    AllTimePowerBestsResponse,
    FtpEstimateResponse,
    PowerBestEntry,
    PowerModelFit,
    PowerModelPoint,
    PowerModelsResponse,
)
from backend.app.services.weight import effective_weight_for, load_weight_log, w_per_kg
from openkoutsi.training_math import (
    CP3_FIT_DURATIONS,
    CP_FIT_DURATIONS,
    EXP_FIT_DURATIONS,
    MODEL_CURVE_DURATIONS,
    POTENTIAL_DURATIONS,
    POWER_BEST_DURATIONS,
    POWER_LAW_FIT_DURATIONS,
    estimate_cp3,
    estimate_cp_wprime,
    estimate_exponential,
    estimate_ftp_simple,
    estimate_power_law,
    model_rmse,
    predict_power,
    sample_power_curve,
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


# Durations each model was fit over (for RMSE) and the shortest duration we are
# willing to plot / predict it at.  Unbounded models (cp2, power_law) are not
# extrapolated below their fit window where they blow up; the CP-anchored models
# (cp3, exp) are bounded at t→0 and safe to sample down to the curve minimum.
_MODEL_FIT_DURATIONS: dict[str, list[int]] = {
    "cp2": CP_FIT_DURATIONS,
    "cp3": CP3_FIT_DURATIONS,
    "exp": EXP_FIT_DURATIONS,
    "power_law": POWER_LAW_FIT_DURATIONS,
}
_MODEL_FLOOR: dict[str, int] = {"cp2": 120, "cp3": 0, "exp": 0, "power_law": 60}


def _build_model_fit(
    model: str, params: tuple[float, ...], rank1: dict[int, float], **fields
) -> PowerModelFit:
    """Assemble a PowerModelFit: sampled curve, profile predictions and RMSE."""
    max_duration = max(rank1) if rank1 else 0
    start = max(5, _MODEL_FLOOR[model])
    curve_durations = [d for d in MODEL_CURVE_DURATIONS if start <= d <= max_duration]

    curve = [
        PowerModelPoint(duration_s=d, power_w=round(p, 1))
        for d, p in sample_power_curve(model, params, curve_durations)
    ]
    predictions = [
        PowerModelPoint(duration_s=d, power_w=round(predict_power(model, params, d), 1))
        for d in POTENTIAL_DURATIONS
        if d >= _MODEL_FLOOR[model]
    ]
    rmse = model_rmse(model, params, rank1, _MODEL_FIT_DURATIONS[model])

    return PowerModelFit(
        model=model,
        available=True,
        rmse=round(rmse, 1) if rmse is not None else None,
        curve=curve,
        predictions=predictions,
        **fields,
    )


def build_power_models(rank1: dict[int, float]) -> list[PowerModelFit]:
    """Fit every power–duration model to the rank-1 bests and assemble results.

    Always returns one entry per model (in a stable order); models that cannot
    be fit from the available data are returned with ``available=False``.
    """
    fits: list[PowerModelFit] = []

    cp, w_prime = estimate_cp_wprime(rank1)
    if cp is not None and w_prime is not None:
        fits.append(_build_model_fit(
            "cp2", (cp, w_prime), rank1,
            cp=round(cp, 1), w_prime=round(w_prime),
        ))
    else:
        fits.append(PowerModelFit(model="cp2"))

    cp3 = estimate_cp3(rank1)
    if cp3 is not None:
        cp_v, wp_v, k_v, pmax_v = cp3
        fits.append(_build_model_fit(
            "cp3", cp3, rank1,
            cp=round(cp_v, 1), w_prime=round(wp_v), k=round(k_v, 2), pmax=round(pmax_v, 1),
        ))
    else:
        fits.append(PowerModelFit(model="cp3"))

    exp = estimate_exponential(rank1)
    if exp is not None:
        cp_v, pmax_v, tau_v = exp
        fits.append(_build_model_fit(
            "exp", exp, rank1,
            cp=round(cp_v, 1), pmax=round(pmax_v, 1), tau=round(tau_v, 1),
        ))
    else:
        fits.append(PowerModelFit(model="exp"))

    power_law = estimate_power_law(rank1)
    if power_law is not None:
        a_v, b_v = power_law
        fits.append(_build_model_fit(
            "power_law", power_law, rank1,
            a=round(a_v, 3), b=round(b_v, 4),
        ))
    else:
        fits.append(PowerModelFit(model="power_law"))

    return fits


@router.get("/power-models", response_model=PowerModelsResponse,
            operation_id="getPowerModels", summary="Fitted power–duration models")
async def get_power_models(
    days: Optional[int] = Query(None, ge=1, description="Fit from bests in the past N days. Omit for all-time."),
    ctx_session=Depends(get_ctx_and_session),
):
    """
    Fit several power–duration models to the athlete's power curve and return,
    for each model: its fitted parameters, a sampled curve for plotting, the
    modeled potential at key durations (5 s / 60 s / 5 min / 20 min), and the
    fit error (RMSE) against the actual bests.

    Models: 2-parameter Critical Power (``cp2``), 3-parameter CP (``cp3``),
    CP-anchored exponential (``exp``) and power law / Riegel (``power_law``).
    All use the rank-1 (single best) power per duration.  Pass ?days=90/180/365
    to fit from a rolling window; omit for all-time.
    """
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=days)
        if days is not None
        else None
    )

    where_clauses = [ActivityPowerBest.athlete_id == athlete.id]
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

    return PowerModelsResponse(models=build_power_models(rank1), days=days)
