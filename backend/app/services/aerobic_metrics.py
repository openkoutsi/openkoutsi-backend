"""Aerobic response metrics applied to an activity (issue #37).

Both the FIT processor and the reprocess endpoint need to derive the same
things from an activity's streams — the decoupling figure (or the reason one
would mislead), the CP/W' snapshot, and the ``w_bal`` stream. This module holds
that step once so the two paths can't drift apart.

Efficiency factor and variability index are not here: they are pure ratios of
columns already on the row and are derived on read in the response schema, so
every activity has them without a reprocess.
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.user_orm import Activity, Athlete
from backend.app.services.power_profile import cp_wprime_as_of
from openkoutsi.training_math import (
    aerobic_decoupling,
    decoupling_unavailable_reason,
    variability_index,
    w_bal_stream,
)


async def apply_aerobic_metrics(
    activity: Activity,
    athlete: Athlete,
    stream_map: dict[str, list],
    session: AsyncSession,
) -> list[float]:
    """Set the aerobic columns on ``activity`` and return its W' balance stream.

    Sets ``decoupling_pct``/``decoupling_reason`` and the ``cp_w``/``w_prime_j``
    snapshot. The returned stream is empty when there is no power or CP could
    not be fit; the caller decides how to persist it (a fresh row on first
    processing, a delete-then-insert on reprocess).

    Must be called *after* the activity's own ``ActivityPowerBest`` rows have
    been added to the session: SQLAlchemy autoflush means the CP fit then sees
    them, so a ride's own efforts count toward the power profile it is judged
    against.
    """
    power: list[float] = [float(v) for v in (stream_map.get("power") or [])]
    heartrate: list[float] = [float(v) for v in (stream_map.get("heartrate") or [])]

    vi = variability_index(activity.weighted_power, activity.avg_power)
    reason = decoupling_unavailable_reason(
        activity.duration_s, power, heartrate, activity.workout_category, vi
    )
    decoupling = None if reason else aerobic_decoupling(power, heartrate)
    if decoupling is None and reason is None:
        # The gate passed but the halves still produced nothing usable.
        reason = "degenerate_hr"

    activity.decoupling_pct = round(decoupling, 2) if decoupling is not None else None
    activity.decoupling_reason = reason

    if not power:
        activity.cp_w = None
        activity.w_prime_j = None
        return []

    cp, w_prime = await cp_wprime_as_of(athlete.id, session, activity.start_time)
    activity.cp_w = round(cp, 1) if cp is not None else None
    activity.w_prime_j = round(w_prime) if w_prime is not None else None

    # Whole joules: a three-hour ride is ~10 800 samples and the sub-joule
    # digits carry no information a rider can act on.
    return [round(v) for v in w_bal_stream(power, cp, w_prime)]
