"""Body-weight history helpers.

The athlete's weight over time lives in the ``WeightLog`` table (one entry per
day, written whenever the profile weight is edited).  Several features need the
*effective* weight at the time of a past activity — the most recent logged
weight on or before that activity's date.  This module is the single source of
truth for that lookup, plus a pass that (re)derives the stored W/kg on every
power-best row when the weight history changes.
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


async def recompute_power_best_weights(athlete_id: str, session: AsyncSession) -> None:
    """Re-derive weight_kg / w_per_kg on every power-best row for an athlete.

    Effective weight depends only on the weight log (not on other activities),
    so this is order-independent and safe after a reverse-chronological mass
    import.  Call it after a bulk sync and whenever the weight history changes,
    since editing/adding a log entry shifts the effective weight for a range of
    past activities.  The caller is responsible for committing the session.
    """
    weight_log = await load_weight_log(athlete_id, session)
    rows = await session.execute(
        select(ActivityPowerBest).where(ActivityPowerBest.athlete_id == athlete_id)
    )
    for best in rows.scalars():
        act_date = best.activity_start_time.date() if best.activity_start_time else None
        weight = effective_weight_for(weight_log, act_date)
        best.weight_kg = weight
        best.w_per_kg = w_per_kg(best.power_w, weight)
