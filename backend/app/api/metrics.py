from datetime import date, datetime, time, timedelta
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.deps import get_ctx_and_session
from backend.app.db.team_session import get_team_session_factory
from backend.app.models.team_orm import Activity, ActivityStream, Athlete, DailyMetric
from backend.app.schemas.metrics import (
    ActivitySummaryResponse,
    FitnessCurrentResponse,
    FitnessMetricResponse,
)
from backend.app.services.metrics_engine import catch_up_metrics
from openkoutsi.sport_matching import CYCLING_SPORT_TYPES
from openkoutsi.zones import Zones

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
            date=today, ctl=0.0, atl=0.0, tsb=0.0, tss_day=0.0
        )
    return FitnessCurrentResponse.model_validate(metric)


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

    if not athlete.hr_zones and not athlete.power_zones:
        raise HTTPException(status_code=400, detail="No zones configured on athlete")

    streams_result = await session.execute(
        select(ActivityStream).where(ActivityStream.activity_id == activity_id)
    )
    streams = {s.stream_type: s.data for s in streams_result.scalars()}

    result: dict = {}

    if athlete.hr_zones and streams.get("heartrate"):
        hr_zones = Zones(*[(z["low"], z["high"]) for z in athlete.hr_zones])
        time_in_hr: dict[str, int] = {}
        for v in streams["heartrate"]:
            zone_i = hr_zones.getZone(int(v))
            name = athlete.hr_zones[zone_i].get("name", f"Z{zone_i + 1}")
            time_in_hr[name] = time_in_hr.get(name, 0) + 1
        result["hr"] = time_in_hr

    if athlete.power_zones and streams.get("power"):
        pw_zones = Zones(*[(z["low"], z["high"]) for z in athlete.power_zones])
        time_in_pw: dict[str, int] = {}
        for v in streams["power"]:
            zone_i = pw_zones.getZone(int(v))
            name = athlete.power_zones[zone_i].get("name", f"Z{zone_i + 1}")
            time_in_pw[name] = time_in_pw.get(name, 0) + 1
        result["power"] = time_in_pw

    return result


@router.get("/ftp-history")
async def get_ftp_history(ctx_session=Depends(get_ctx_and_session)):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    return athlete.ftp_tests or []


@router.post("/catch-up", status_code=200)
async def catch_up(ctx_session=Depends(get_ctx_and_session)):
    """Fill missing DailyMetric rows using stored TSS. Called on dashboard load."""
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
    Recompute TSS for every processed activity using the athlete's current FTP/max_hr,
    then rebuild CTL/ATL/TSB from the earliest activity forward.

    Returns immediately (202); work happens in the background.
    """
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    background_tasks.add_task(_bg_full_recalculate, ctx.team_id, athlete.id)
    return {"status": "recalculation started"}


_RECALCULATE_LOOKBACK_DAYS = 180


async def _bg_full_recalculate(team_id: str, athlete_id: str) -> None:
    from sqlalchemy import delete
    from openkoutsi.training_math import normalized_power, calculate_tss, compute_power_bests, compute_distance_bests
    from backend.app.services.metrics_engine import recalculate_from
    from backend.app.models.team_orm import ActivityDistanceBest, ActivityPowerBest

    lookback_date = date.today() - timedelta(days=_RECALCULATE_LOOKBACK_DAYS)

    async with get_team_session_factory(team_id)() as session:
        athlete_result = await session.execute(
            select(Athlete).where(Athlete.id == athlete_id)
        )
        athlete = athlete_result.scalar_one()

        # Load recent processed activities only; CTL/ATL seed error < 2% after 180 days
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
            # Re-derive NP from stored power stream (if any)
            stream_result = await session.execute(
                select(ActivityStream).where(
                    ActivityStream.activity_id == activity.id,
                    ActivityStream.stream_type == "power",
                )
            )
            power_stream = stream_result.scalar_one_or_none()
            power_data: list[float] = power_stream.data if power_stream else []

            np = (
                normalized_power(power_data)
                if len(power_data) >= 30
                else (activity.avg_power)
            )

            tss, intensity_factor = calculate_tss(
                activity.duration_s or 0,
                np,
                activity.avg_hr,
                athlete.ftp,
                athlete.max_hr,
            )

            activity.tss = tss
            activity.intensity_factor = intensity_factor
            if np is not None:
                activity.normalized_power = np

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
