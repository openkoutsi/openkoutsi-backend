from datetime import date, datetime, time, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.user_orm import Activity, DailyMetric
from openkoutsi.fatigue_metrics import compute_daily_metrics

# How far back to scan for stale metrics caused by deleted activities.
_STALE_CHECK_DAYS = 90


async def _find_stale_from(
    athlete_id: str, today: date, session: AsyncSession
) -> date | None:
    """Return the earliest date where DailyMetric.load_day doesn't match the
    sum of Activity.load for that day, or None if everything is consistent.

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
    stored = {m.date: m.load_day for m in metrics_result.scalars()}
    if not stored:
        return None

    cutoff = datetime.combine(lookback, time.min)
    acts_result = await session.execute(
        select(Activity).where(
            Activity.athlete_id == athlete_id,
            Activity.start_time >= cutoff,
            Activity.load.is_not(None),
            Activity.status == "processed",
        )
    )
    actual: dict[date, float] = {}
    for act in acts_result.scalars():
        if act.start_time is None:
            continue
        day = act.start_time.date() if hasattr(act.start_time, "date") else act.start_time
        actual[day] = actual.get(day, 0.0) + (act.load or 0.0)

    earliest: date | None = None
    for day, stored_tss in stored.items():
        if abs(stored_tss - actual.get(day, 0.0)) > 0.01:
            if earliest is None or day < earliest:
                earliest = day
    return earliest


async def _find_first_missing_day(
    athlete_id: str, session: AsyncSession
) -> date | None:
    """Return the earliest day with no DailyMetric row between the athlete's
    first and last stored days, or None if the rows are contiguous.

    A hole is what an older ``recalculate_from`` left when the day before its
    start had no row: it seeded 0.0/0.0, so every day after the hole was
    computed from zero. Recalculating from the hole back-fills it and restores
    them. Unlike the stale-Load check this covers the whole history — one
    aggregate over the primary key answers "any holes?", and the dates are
    only listed when there is one.
    """
    first, last, count = (
        await session.execute(
            select(
                func.min(DailyMetric.date),
                func.max(DailyMetric.date),
                func.count(),
            ).where(DailyMetric.athlete_id == athlete_id)
        )
    ).one()
    if first is None or (last - first).days + 1 == count:
        return None

    dates = await session.execute(
        select(DailyMetric.date)
        .where(DailyMetric.athlete_id == athlete_id)
        .order_by(DailyMetric.date)
    )
    expected = first
    for day in dates.scalars():
        if day != expected:
            return expected
        expected = day + timedelta(days=1)
    return None


async def catch_up_metrics(athlete_id: str, session: AsyncSession) -> bool:
    """Fill missing DailyMetric rows up to today, back-fill any day missing
    from the athlete's history, and fix any rows made stale by deleted
    activities.

    Returns True if rows were written or corrected, False if already up to date.
    No stream reprocessing — uses stored Load values only.
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

    for candidate in (
        await _find_first_missing_day(athlete_id, session),
        await _find_stale_from(athlete_id, today, session),
    ):
        if candidate is not None and (recalc_from is None or candidate < recalc_from):
            recalc_from = candidate

    if recalc_from is not None:
        await recalculate_from(athlete_id, recalc_from, session)
        return True
    return False


async def recalculate_from(
    athlete_id: str, from_date: date, session: AsyncSession
) -> None:
    # Seed Fitness/Fatigue from the latest stored day before from_date. That is
    # usually the day before, but not always: rows are written only up to the
    # "today" of whatever last ran, so a day with no ride and no dashboard visit
    # leaves a hole. Seeding 0.0 across that hole reset the athlete's history to
    # zero; instead the walk starts the day after the seed, filling the hole on
    # the way. 0.0 is only the seed when there is no earlier row at all.
    prev_result = await session.execute(
        select(DailyMetric)
        .where(
            DailyMetric.athlete_id == athlete_id,
            DailyMetric.date < from_date,
        )
        .order_by(DailyMetric.date.desc())
        .limit(1)
    )
    prev = prev_result.scalar_one_or_none()
    if prev is not None:
        from_date = prev.date + timedelta(days=1)
    initial_fitness = prev.fitness if prev else 0.0
    initial_fatigue = prev.fatigue if prev else 0.0

    # Bucket Load by date for all processed activities from from_date onwards
    cutoff = datetime.combine(from_date, time.min)
    acts_result = await session.execute(
        select(Activity).where(
            Activity.athlete_id == athlete_id,
            Activity.start_time >= cutoff,
            Activity.load.is_not(None),
            Activity.status == "processed",
        )
    )
    load_by_date: dict[date, float] = {}
    for act in acts_result.scalars():
        if act.start_time is None:
            continue
        day = act.start_time.date() if hasattr(act.start_time, "date") else act.start_time
        load_by_date[day] = load_by_date.get(day, 0.0) + (act.load or 0.0)

    metrics = compute_daily_metrics(load_by_date, from_date, date.today(), initial_fitness, initial_fatigue)

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

        metric.fitness = m["fitness"]
        metric.fatigue = m["fatigue"]
        metric.form = m["form"]
        metric.load_day = m["load_day"]

    await session.commit()
