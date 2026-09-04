"""Body-weight history helpers.

The athlete's weight over time lives in the ``WeightLog`` table (one entry per
day, written whenever the profile weight is edited).  Several features need the
*effective* weight at the time of a past activity — the most recent logged
weight on or before that activity's date.  This module is the single source of
truth for that lookup, plus a pass that fills in a power best's weight when it
was stored without one.

Weight on a power best is a **snapshot**, captured when the activity is processed
and never rewritten. A new weigh-in therefore applies only from that day onward —
an older activity keeps whatever the athlete weighed at the time, or no weight at
all. (Same reasoning as the frozen ``Activity.zone_times``: editing today's
profile must not silently rewrite past rides.)
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.user_orm import ActivityPowerBest, WeightLog


async def load_weight_log(athlete_id: str, session: AsyncSession) -> list[tuple[date, float]]:
    """Return the athlete's weight log as (effective_date, weight_kg) ascending by date."""
    rows = await session.execute(
        select(WeightLog)
        .where(WeightLog.athlete_id == athlete_id)
        .order_by(WeightLog.effective_date)
    )
    return [(w.effective_date, w.weight_kg) for w in rows.scalars().all()]


def effective_weight_for(
    weight_log: list[tuple[date, float]],
    activity_date: Optional[date],
) -> Optional[float]:
    """Most recent logged weight whose effective_date <= activity_date.

    Returns ``None`` when the date is unknown or no entry predates it — callers
    treat a missing weight as "no W/kg" rather than back-attributing a later one.
    ``weight_log`` must be sorted ascending by date (see :func:`load_weight_log`).
    """
    if not activity_date or not weight_log:
        return None
    result: Optional[float] = None
    for eff_date, w_kg in weight_log:
        if eff_date <= activity_date:
            result = w_kg
        else:
            break
    return result


def w_per_kg(power_w: Optional[float], weight_kg: Optional[float]) -> Optional[float]:
    """Watts per kilogram, or ``None`` when weight is unknown/non-positive."""
    if power_w is None or not weight_kg or weight_kg <= 0:
        return None
    return power_w / weight_kg


async def backfill_missing_power_best_weights(athlete_id: str, session: AsyncSession) -> None:
    """Fill weight_kg / w_per_kg on power-best rows that were stored without one.

    Only rows with no weight yet are touched, and only when the log holds an
    entry on or before the activity's date — a row that already carries a weight
    keeps it, and a row with no contemporaneous weight stays empty.  So this can
    never rewrite an activity's W/kg history; it only repairs rows created before
    the weight was known (a reverse-chronological mass import, or a first
    weigh-in recorded later the same day).

    Effective weight depends only on the weight log (not on other activities), so
    this is order-independent.  The caller commits the session.
    """
    weight_log = await load_weight_log(athlete_id, session)
    if not weight_log:
        return
    rows = await session.execute(
        select(ActivityPowerBest).where(
            ActivityPowerBest.athlete_id == athlete_id,
            ActivityPowerBest.weight_kg.is_(None),
        )
    )
    for best in rows.scalars():
        act_date = best.activity_start_time.date() if best.activity_start_time else None
        weight = effective_weight_for(weight_log, act_date)
        if weight is None:
            continue
        best.weight_kg = weight
        best.w_per_kg = w_per_kg(best.power_w, weight)
