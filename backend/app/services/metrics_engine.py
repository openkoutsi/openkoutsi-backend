from datetime import date, datetime, time, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.team_orm import Activity, DailyMetric
from openkoutsi.fatigue_metrics import compute_daily_metrics

# How far back to scan for stale metrics caused by deleted activities.
_STALE_CHECK_DAYS = 90


async def _find_stale_from(
    athlete_id: str, today: date, session: AsyncSession
) -> date | None:
    """Return the earliest date where DailyMetric.tss_day doesn't match the
    sum of Activity.tss for that day, or None if everything is consistent.

    A mismatch indicates activities were deleted (or added) without triggering
    a metric recalculation — e.g. via direct DB cleanup or dedup tooling.
    """
    lookback = today - timedelta(days=_STALE_CHECK_DAYS)

    metrics_result = await session.execute(
        select(DailyMetric).where(
            DailyMetric.athlete_id == athlete_id,
            DailyMetric.date >= lookback,
        )
    )
    stored = {m.date: m.tss_day for m in metrics_result.scalars()}
    if not stored:
        return None

    cutoff = datetime.combine(lookback, time.min)
    acts_result = await session.execute(
        select(Activity).where(
            Activity.athlete_id == athlete_id,
            Activity.start_time >= cutoff,
            Activity.tss.is_not(None),
            Activity.status == "processed",
        )
    )
    actual: dict[date, float] = {}
    for act in acts_result.scalars():
        if act.start_time is None:
            continue
        day = act.start_time.date() if hasattr(act.start_time, "date") else act.start_time
        actual[day] = actual.get(day, 0.0) + (act.tss or 0.0)

    earliest: date | None = None
    for day, stored_tss in stored.items():
        if abs(stored_tss - actual.get(day, 0.0)) > 0.01:
            if earliest is None or day < earliest:
                earliest = day
    return earliest


async def catch_up_metrics(athlete_id: str, session: AsyncSession) -> bool:
    """Fill missing DailyMetric rows up to today and fix any rows made stale
    by deleted activities.

    Returns True if rows were written or corrected, False if already up to date.
    No stream reprocessing — uses stored TSS values only.
    """
    today = date.today()
    recalc_from: date | None = None

    existing = await session.execute(
        select(DailyMetric).where(
            DailyMetric.athlete_id == athlete_id,
            DailyMetric.date == today,
        )
    )
    if existing.scalar_one_or_none() is None:
        last = await session.execute(
            select(DailyMetric)
            .where(DailyMetric.athlete_id == athlete_id)
            .order_by(DailyMetric.date.desc())
            .limit(1)
        )
        last_metric = last.scalar_one_or_none()
        recalc_from = (last_metric.date + timedelta(days=1)) if last_metric else today

    stale_from = await _find_stale_from(athlete_id, today, session)
    if stale_from is not None:
        if recalc_from is None or stale_from < recalc_from:
            recalc_from = stale_from

    if recalc_from is not None:
        await recalculate_from(athlete_id, recalc_from, session)
        return True
    return False


async def recalculate_from(
    athlete_id: str, from_date: date, session: AsyncSession
) -> None:
    # Seed CTL/ATL from the day before from_date (or 0.0)
    prev_date = from_date - timedelta(days=1)
    prev_result = await session.execute(
        select(DailyMetric).where(
            DailyMetric.athlete_id == athlete_id,
            DailyMetric.date == prev_date,
        )
    )
    prev = prev_result.scalar_one_or_none()
    initial_ctl = prev.ctl if prev else 0.0
    initial_atl = prev.atl if prev else 0.0

    # Bucket TSS by date for all processed activities from from_date onwards
    cutoff = datetime.combine(from_date, time.min)
    acts_result = await session.execute(
        select(Activity).where(
            Activity.athlete_id == athlete_id,
            Activity.start_time >= cutoff,
            Activity.tss.is_not(None),
            Activity.status == "processed",
        )
    )
    tss_by_date: dict[date, float] = {}
    for act in acts_result.scalars():
        if act.start_time is None:
            continue
        day = act.start_time.date() if hasattr(act.start_time, "date") else act.start_time
        tss_by_date[day] = tss_by_date.get(day, 0.0) + (act.tss or 0.0)

    metrics = compute_daily_metrics(tss_by_date, from_date, date.today(), initial_ctl, initial_atl)

    for m in metrics:
        existing = await session.execute(
            select(DailyMetric).where(
                DailyMetric.athlete_id == athlete_id,
                DailyMetric.date == m["date"],
            )
        )
        metric = existing.scalar_one_or_none()
        if metric is None:
            metric = DailyMetric(athlete_id=athlete_id, date=m["date"])
            session.add(metric)

        metric.ctl = m["ctl"]
        metric.atl = m["atl"]
        metric.tsb = m["tsb"]
        metric.tss_day = m["tss_day"]

    await session.commit()
