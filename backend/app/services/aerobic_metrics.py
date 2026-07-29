"""Aerobic response metrics applied to an activity (issue #37).

Four paths populate an activity from its streams — ``process_fit_file``, the
reprocess endpoint, and both provider-sync paths (FIT download and the
stream-based Strava fallback). All of them need the same derivations: the
decoupling figure (or the reason one would mislead), the CP/W' snapshot, and the
``w_bal`` stream. This module holds that step once so the four can't drift
apart, and ``tests/integration/test_writer_paths.py`` asserts the invariant
across all of them so a fifth path fails loudly rather than silently shipping
nulls.

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

# W' balance arithmetic is joules per sample and only equals joules per second at
# 1 Hz. A stream carrying far fewer samples than the ride has elapsed seconds was
# either recorded at a lower rate (Garmin smart recording) or is too gappy to
# integrate, and the depletion rate would be wrong by the sampling ratio.
#
# The threshold is deliberately loose: a genuine 1 Hz ride with long stops also
# has fewer samples than elapsed seconds, and a tight bound would quietly remove
# the feature from anyone who stops at a café.
MIN_SAMPLE_COVERAGE = 0.5


async def apply_aerobic_metrics(
    activity: Activity,
    athlete: Athlete,
    stream_map: dict[str, list],
    session: AsyncSession,
) -> list[float]:
    """Set the aerobic columns on ``activity`` and return its W' balance stream.

    Sets ``decoupling_pct``/``decoupling_reason`` and the
    ``cp_w``/``w_prime_j``/``cp_fit_points`` snapshot. Exactly one of
    ``decoupling_pct`` and ``decoupling_reason`` is always set. The returned
    stream is empty when there is no usable power, when CP could not be fit, or
    when the sampling rate can't support the integration; the caller decides how
    to persist it (a fresh row on first processing, a delete-then-insert on
    reprocess).

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
        # Defensive only. The gate checks both halves for usable data, so a
        # passing gate should always yield a number; this keeps the
        # exactly-one-of-two invariant true even if that ever stops holding.
        reason = "degenerate_hr"

    activity.decoupling_pct = round(decoupling, 2) if decoupling is not None else None
    activity.decoupling_reason = reason

    activity.cp_w = None
    activity.w_prime_j = None
    activity.cp_fit_points = None

    if not any(v > 0 for v in power):
        return []

    # A dateless activity can't be fit "as of" anything, and falling through to
    # an all-time fit would silently produce the exact anachronism this snapshot
    # exists to prevent. Writing nothing is the conservative outcome.
    if activity.start_time is None:
        return []

    cp, w_prime, fit_points = await cp_wprime_as_of(
        athlete.id, session, activity.start_time
    )
    activity.cp_w = round(cp, 1) if cp is not None else None
    activity.w_prime_j = round(w_prime) if w_prime is not None else None
    activity.cp_fit_points = fit_points

    if not _sampling_supports_integration(power, activity.duration_s):
        return []

    # Whole joules: a three-hour ride is ~10 800 samples and the sub-joule
    # digits carry no information a rider can act on.
    return [round(v) for v in w_bal_stream(power, cp, w_prime)]


def _sampling_supports_integration(power: list[float], duration_s: int | None) -> bool:
    """Is this power stream dense enough to integrate as one sample per second?"""
    if not duration_s or duration_s <= 0:
        # No elapsed time to compare against — nothing to contradict 1 Hz.
        return True
    return len(power) >= MIN_SAMPLE_COVERAGE * duration_s
