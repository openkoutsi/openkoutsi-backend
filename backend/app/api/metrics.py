from datetime import date, datetime, time, timedelta
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.deps import get_ctx_and_session
from backend.app.db.user_session import get_user_session_factory
from backend.app.models.user_orm import Activity, ActivityStream, Athlete, DailyMetric
from backend.app.schemas.metrics import (
    ActivitySummaryResponse,
    EfficiencyPoint,
    FitnessCurrentResponse,
    FitnessForecastResponse,
    FitnessMetricResponse,
    IntensityBasis,
    IntensityDistributionResponse,
    IntensityMethod,
    WeeklyZoneBucket,
)
from backend.app.services.intensity_distribution import (
    backfill_window_snapshots,
    compute_intensity_distribution,
    resolve_window,
)
from backend.app.services.metrics_engine import catch_up_metrics
from backend.app.services.metrics_forecast import (
    DEFAULT_FORECAST_DAYS,
    MAX_FORECAST_DAYS,
    forecast_fitness,
)
from backend.app.services.zone_times import compute_zone_times, ensure_zone_times
from openkoutsi.sport_matching import CYCLING_SPORT_TYPES
from openkoutsi.training_math import (
    DECOUPLING_EXCLUDED_CATEGORIES,
    DECOUPLING_MAX_VI,
    efficiency_factor,
    variability_index,
)

router = APIRouter(prefix="/metrics", tags=["metrics"])


async def _get_athlete(global_user_id: str, session: AsyncSession) -> Athlete:
    result = await session.execute(select(Athlete).where(Athlete.global_user_id == global_user_id))
    athlete = result.scalar_one_or_none()
    if athlete is None:
        raise HTTPException(status_code=404, detail="Athlete profile not found")
    return athlete


@router.get("/fitness", response_model=list[FitnessMetricResponse])
async def get_fitness(
    start: Optional[date] = Query(None),
    end: Optional[date] = Query(None),
    days: Optional[int] = Query(None, ge=1, le=3650),
    ctx_session=Depends(get_ctx_and_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    query = select(DailyMetric).where(DailyMetric.athlete_id == athlete.id)

    if days is not None and start is None:
        start = date.today() - timedelta(days=days)

    if start:
        query = query.where(DailyMetric.date >= start)
    if end:
        query = query.where(DailyMetric.date <= end)

    result = await session.execute(query.order_by(DailyMetric.date))
    return [FitnessMetricResponse.model_validate(m) for m in result.scalars().all()]


@router.get("/activity-summary", response_model=ActivitySummaryResponse)
async def get_activity_summary(
    start: Optional[date] = Query(None),
    end: Optional[date] = Query(None),
    days: Optional[int] = Query(None, ge=1, le=3650),
    ctx_session=Depends(get_ctx_and_session),
):
    """Totals (count, active time, distance) for cycling activities in a period."""
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    if days is not None and start is None:
        start = date.today() - timedelta(days=days)

    query = select(
        func.count(Activity.id),
        func.coalesce(func.sum(Activity.duration_s), 0),
        func.coalesce(func.sum(Activity.distance_m), 0.0),
    ).where(
        Activity.athlete_id == athlete.id,
        Activity.sport_type.in_(CYCLING_SPORT_TYPES),
    )

    if start:
        query = query.where(Activity.start_time >= datetime.combine(start, time.min))
    if end:
        query = query.where(Activity.start_time <= datetime.combine(end, time.max))

    num, total_duration, total_distance = (await session.execute(query)).one()
    return ActivitySummaryResponse(
        num_activities=num,
        total_duration_s=int(total_duration),
        total_distance_m=float(total_distance),
    )


@router.get("/fitness/current", response_model=FitnessCurrentResponse)
async def get_fitness_current(ctx_session=Depends(get_ctx_and_session)):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    today = date.today()

    result = await session.execute(
        select(DailyMetric).where(
            DailyMetric.athlete_id == athlete.id,
            DailyMetric.date == today,
        )
    )
    metric = result.scalar_one_or_none()
    if metric is None:
        fallback = await session.execute(
            select(DailyMetric)
            .where(DailyMetric.athlete_id == athlete.id)
            .order_by(DailyMetric.date.desc())
            .limit(1)
        )
        metric = fallback.scalar_one_or_none()
    if metric is None:
        return FitnessCurrentResponse(
            date=today, fitness=0.0, fatigue=0.0, form=0.0, load_day=0.0
        )
    return FitnessCurrentResponse.model_validate(metric)


@router.get("/fitness/forecast", response_model=list[FitnessForecastResponse])
async def get_fitness_forecast(
    days: int = Query(DEFAULT_FORECAST_DAYS, ge=1, le=MAX_FORECAST_DAYS),
    ctx_session=Depends(get_ctx_and_session),
):
    """Project Fitness/Fatigue/Form forward from the athlete's planned workouts.

    Covers ``[today + 1, today + days]``. The projection itself is never stored,
    so it always reflects the plans as they stand right now. Kept separate from
    ``GET /metrics/fitness`` so the historical contract is untouched and the
    caller decides whether to concatenate the two series.

    Metrics are caught up first, exactly as the dashboard does via
    ``POST /metrics/catch-up``, so the projection is always seeded from today
    rather than from whatever the last stored day happened to be. Without it the
    answer would depend on whether the client had hit catch-up beforehand.
    """
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    await catch_up_metrics(athlete.id, session)
    rows = await forecast_fitness(athlete.id, session, days=days)
    return [FitnessForecastResponse(**row) for row in rows]


# Efficiency factor over a very short ride is noise; require a reasonable block
# of riding before a point joins the trend.
EFFICIENCY_MIN_DURATION_S = 1200

# Hard cap on points returned. This is a trend chart — nobody scrolls back a
# decade in it — and an uncapped all-time query on a long-tenured athlete is
# unbounded work per request. The most recent rides are the ones that matter, so
# the cap takes from the newest end and the result is reversed for display.
EFFICIENCY_MAX_POINTS = 1000


@router.get("/efficiency", response_model=list[EfficiencyPoint],
            operation_id="getEfficiencyTrend", summary="Aerobic efficiency trend")
async def get_efficiency_trend(
    days: Optional[int] = Query(None, ge=1, le=3650, description="Restrict to rides from the past N days. Omit for all-time."),
    ctx_session=Depends(get_ctx_and_session),
):
    """
    Aerobic efficiency (weighted power per heartbeat) for each steady endurance
    ride, oldest first — the trend that shows aerobic progress a rising Fitness
    number alone doesn't.

    Restricted to cycling rides of at least 20 minutes that have both weighted
    power and average heart rate, and that were steady enough for the number to
    mean something: the same "not an interval session" test the decoupling gate
    uses (variability index at or below 1.10, and not categorized as VO2max,
    anaerobic or sprint work). Each ride's decoupling figure rides along when it
    has one.

    Computed on read from stored columns — nothing extra is persisted, so the
    trend covers the athlete's whole history immediately. Capped at the most
    recent 1000 qualifying rides.
    """
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    # Select the columns rather than the entity: this reads seven fields, and
    # hydrating full ORM rows would deserialise each activity's `zone_times`
    # JSON and every other column for nothing.
    query = select(
        Activity.id,
        Activity.start_time,
        Activity.duration_s,
        Activity.weighted_power,
        Activity.avg_power,
        Activity.avg_hr,
        Activity.decoupling_pct,
    ).where(
        Activity.athlete_id == athlete.id,
        Activity.sport_type.in_(CYCLING_SPORT_TYPES),
        Activity.start_time.is_not(None),
        Activity.weighted_power.is_not(None),
        Activity.avg_hr.is_not(None),
        Activity.avg_power.is_not(None),
        Activity.duration_s >= EFFICIENCY_MIN_DURATION_S,
        # NOT IN is NULL for an uncategorized ride, so spell out the null case.
        Activity.workout_category.not_in(sorted(DECOUPLING_EXCLUDED_CATEGORIES))
        | Activity.workout_category.is_(None),
    )
    if days is not None:
        cutoff = datetime.combine(date.today() - timedelta(days=days), time.min)
        query = query.where(Activity.start_time >= cutoff)

    # Take the cap from the newest end, then flip to the oldest-first order the
    # chart wants — truncating the recent end would be the wrong half to lose.
    result = await session.execute(
        query.order_by(Activity.start_time.desc()).limit(EFFICIENCY_MAX_POINTS)
    )

    points: list[EfficiencyPoint] = []
    for row in result.all():
        vi = variability_index(row.weighted_power, row.avg_power)
        if vi is None or vi > DECOUPLING_MAX_VI:
            continue
        ef = efficiency_factor(row.weighted_power, row.avg_hr)
        if ef is None:
            continue
        points.append(
            EfficiencyPoint(
                activity_id=row.id,
                date=row.start_time.date(),
                duration_s=row.duration_s,
                efficiency_factor=round(ef, 3),
                decoupling_pct=row.decoupling_pct,
            )
        )
    points.reverse()
    return points


@router.get("/zones/weekly", response_model=list[WeeklyZoneBucket])
async def get_zones_weekly(
    start: Optional[date] = Query(None),
    end: Optional[date] = Query(None),
    days: Optional[int] = Query(None, ge=1, le=3650),
    ctx_session=Depends(get_ctx_and_session),
):
    """Accumulated time-in-zone (power + HR) per ISO week over a period.

    Sums each cycling activity's frozen ``zone_times`` snapshot into Monday-based
    weekly buckets. Legacy activities without a snapshot are backfilled on the
    fly (using current zones) and frozen, mirroring the fitness catch-up flow.

    Declared before ``/zones/{activity_id}`` so "weekly" isn't matched as an id.
    """
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    if days is not None and start is None:
        start = date.today() - timedelta(days=days)

    query = select(Activity).where(
        Activity.athlete_id == athlete.id,
        Activity.sport_type.in_(CYCLING_SPORT_TYPES),
        Activity.status == "processed",
        Activity.start_time.is_not(None),
    )
    if start:
        query = query.where(Activity.start_time >= datetime.combine(start, time.min))
    if end:
        query = query.where(Activity.start_time <= datetime.combine(end, time.max))

    activities = (await session.execute(query)).scalars().all()

    if await ensure_zone_times(athlete, session, activities):
        await session.commit()

    buckets: dict[date, dict[str, dict[str, int]]] = {}
    for activity in activities:
        if not activity.zone_times or activity.start_time is None:
            continue
        day = activity.start_time.date()
        week_start = day - timedelta(days=day.weekday())  # Monday
        bucket = buckets.setdefault(week_start, {})
        for kind in ("hr", "power"):
            times = activity.zone_times.get(kind)
            if not times:
                continue
            dest = bucket.setdefault(kind, {})
            for name, seconds in times.items():
                dest[name] = dest.get(name, 0) + seconds

    return [
        WeeklyZoneBucket(
            week_start=week_start,
            hr=data.get("hr", {}),
            power=data.get("power", {}),
        )
        for week_start, data in sorted(buckets.items())
    ]


@router.get(
    "/intensity-distribution",
    response_model=IntensityDistributionResponse,
    operation_id="getIntensityDistribution",
    summary="Intensity distribution over a training block",
)
async def get_intensity_distribution(
    start: Optional[date] = Query(None),
    end: Optional[date] = Query(None),
    days: Optional[int] = Query(None, ge=1, le=3650),
    basis: Optional[IntensityBasis] = Query(None),
    method: IntensityMethod = Query("time"),
    ctx_session=Depends(get_ctx_and_session),
):
    """Three-band intensity distribution and its shape over a period.

    Weekly zones say what last week held; this says what a *block* came out as
    — polarized, pyramidal, threshold-heavy, or almost all easy. Defaults to
    the last 12 weeks.

    ``method=time`` sums the frozen ``zone_times`` snapshots and reports
    percentages of time. ``method=session`` counts each ride whole by its
    workout category and reports percentages of sessions. They disagree by
    design: warm-ups and coast-downs pull the time method toward pyramidal.

    ``basis`` selects power or HR zones, preferring power when both exist; it
    does not apply to ``method=session`` and is echoed as ``null`` there.
    """
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    try:
        start, end = resolve_window(start, end, days)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Freezing missing snapshots is this endpoint's business, not the
    # distribution service's: the service is also called from the plan
    # generator, on a session holding uncommitted deletions that a commit here
    # would make permanent. The write stays where the transaction is owned.
    if method == "time":
        await backfill_window_snapshots(athlete, session, start, end)

    return await compute_intensity_distribution(
        athlete, session, start=start, end=end, basis=basis, method=method
    )


@router.get("/zones/{activity_id}")
async def get_zones(
    activity_id: str,
    ctx_session=Depends(get_ctx_and_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    activity_result = await session.execute(
        select(Activity).where(
            Activity.id == activity_id, Activity.athlete_id == athlete.id
        )
    )
    activity = activity_result.scalar_one_or_none()
    if activity is None:
        raise HTTPException(status_code=404, detail="Activity not found")

    # Prefer the frozen snapshot captured when the activity was processed.
    if activity.zone_times is not None:
        return activity.zone_times

    if not athlete.hr_zones and not athlete.power_zones:
        raise HTTPException(status_code=400, detail="No zones configured on athlete")

    # Legacy activity with no snapshot yet: compute from streams using the
    # current zones and freeze it, so future zone edits leave it untouched.
    streams_result = await session.execute(
        select(ActivityStream).where(ActivityStream.activity_id == activity_id)
    )
    streams = {s.stream_type: s.data for s in streams_result.scalars()}

    zone_times = compute_zone_times(streams, athlete.hr_zones, athlete.power_zones)
    if zone_times is None:
        return {}
    activity.zone_times = zone_times
    await session.commit()
    return zone_times


@router.get("/ftp/history", operation_id="getFtpHistory", summary="FTP history")
async def get_ftp_history(ctx_session=Depends(get_ctx_and_session)):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    return athlete.ftp_tests or []


@router.post("/catch-up", status_code=200)
async def catch_up(ctx_session=Depends(get_ctx_and_session)):
    """Fill missing DailyMetric rows using stored Load. Called on dashboard load."""
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    updated = await catch_up_metrics(athlete.id, session)
    return {"updated": updated}


@router.post("/recalculate", status_code=202)
async def recalculate_all(
    background_tasks: BackgroundTasks,
    ctx_session=Depends(get_ctx_and_session),
):
    """
    Recompute Load for every processed activity using the athlete's current FTP/max_hr,
    then rebuild Fitness/Fatigue/Form from the earliest activity forward.

    Returns immediately (202); work happens in the background.
    """
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    background_tasks.add_task(_bg_full_recalculate, ctx.user_id, athlete.id)
    return {"status": "recalculation started"}


_RECALCULATE_LOOKBACK_DAYS = 180


async def _bg_full_recalculate(user_id: str, athlete_id: str) -> None:
    from sqlalchemy import delete
    from openkoutsi.training_math import weighted_power, calculate_load, compute_power_bests, compute_distance_bests
    from backend.app.services.metrics_engine import recalculate_from
    from backend.app.models.user_orm import ActivityDistanceBest, ActivityPowerBest

    lookback_date = date.today() - timedelta(days=_RECALCULATE_LOOKBACK_DAYS)

    async with get_user_session_factory(user_id)() as session:
        athlete_result = await session.execute(
            select(Athlete).where(Athlete.id == athlete_id)
        )
        athlete = athlete_result.scalar_one()

        # Load recent processed activities only; Fitness/Fatigue seed error < 2% after 180 days
        acts_result = await session.execute(
            select(Activity)
            .where(
                Activity.athlete_id == athlete_id,
                Activity.status == "processed",
                Activity.start_time >= datetime.combine(lookback_date, time.min),
            )
            .order_by(Activity.start_time)
        )
        activities = acts_result.scalars().all()

        if not activities:
            return

        earliest: date | None = None

        for activity in activities:
            # Re-derive Weighted Power from stored power stream (if any)
            stream_result = await session.execute(
                select(ActivityStream).where(
                    ActivityStream.activity_id == activity.id,
                    ActivityStream.stream_type == "power",
                )
            )
            power_stream = stream_result.scalar_one_or_none()
            power_data: list[float] = power_stream.data if power_stream else []

            wp = (
                weighted_power(power_data)
                if len(power_data) >= 30
                else (activity.avg_power)
            )

            load, intensity = calculate_load(
                activity.duration_s or 0,
                wp,
                activity.avg_hr,
                athlete.ftp,
                athlete.max_hr,
            )

            activity.load = load
            activity.intensity = intensity
            if wp is not None:
                activity.weighted_power = wp

            # Recompute power bests from the stored stream
            if power_data:
                await session.execute(
                    delete(ActivityPowerBest).where(
                        ActivityPowerBest.activity_id == activity.id
                    )
                )
                for duration_s, power_w in compute_power_bests(power_data).items():
                    session.add(
                        ActivityPowerBest(
                            activity_id=activity.id,
                            athlete_id=athlete_id,
                            duration_s=duration_s,
                            power_w=power_w,
                            activity_start_time=activity.start_time,
                        )
                    )

            # Recompute distance bests from the stored speed stream
            speed_result = await session.execute(
                select(ActivityStream).where(
                    ActivityStream.activity_id == activity.id,
                    ActivityStream.stream_type == "speed",
                )
            )
            speed_stream_row = speed_result.scalar_one_or_none()
            speed_data: list[float] = speed_stream_row.data if speed_stream_row else []

            if speed_data:
                await session.execute(
                    delete(ActivityDistanceBest).where(
                        ActivityDistanceBest.activity_id == activity.id
                    )
                )
                for distance_m, time_s in compute_distance_bests(speed_data).items():
                    session.add(
                        ActivityDistanceBest(
                            activity_id=activity.id,
                            athlete_id=athlete_id,
                            distance_m=distance_m,
                            time_s=time_s,
                            activity_start_time=activity.start_time,
                        )
                    )

            if activity.start_time is not None:
                day = (
                    activity.start_time.date()
                    if hasattr(activity.start_time, "date")
                    else activity.start_time
                )
                if earliest is None or day < earliest:
                    earliest = day

        await session.commit()

        if earliest is not None:
            await recalculate_from(athlete_id, earliest, session)
