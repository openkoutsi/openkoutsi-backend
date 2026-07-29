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

import uuid
from datetime import datetime, timezone

from sqlalchemy import delete as sa_delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.user_orm import Activity, ActivityStream, Athlete
from backend.app.services.power_profile import (
    cp_fit_bests_ordered,
    cp_wprime_as_of,
    fit_cp_wprime,
)
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


async def replace_w_bal_stream(
    session: AsyncSession, activity_id: str, data: list[float]
) -> None:
    """Set an activity's ``w_bal`` stream to ``data``, removing it when empty.

    Delete-then-insert rather than update: an empty result has to *remove* the
    row, or an activity whose CP fit stopped being usable would keep serving a
    stream computed from numbers its columns no longer carry.
    """
    await session.execute(
        sa_delete(ActivityStream).where(
            ActivityStream.activity_id == activity_id,
            ActivityStream.stream_type == "w_bal",
        )
    )
    if data:
        session.add(
            ActivityStream(
                id=str(uuid.uuid4()),
                activity_id=activity_id,
                stream_type="w_bal",
                data=data,
            )
        )


def _as_utc(value: datetime | None) -> datetime | None:
    """Normalise to tz-aware UTC so DB-returned datetimes compare safely.

    SQLite hands back naive datetimes for ``DateTime(timezone=True)`` columns,
    and comparing one of those against a tz-aware value raises.
    """
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


async def refit_cp_snapshots(
    athlete_id: str, session: AsyncSession, *, since: datetime
) -> int:
    """Re-fit the frozen CP/W' snapshots for activities from ``since`` onward.

    A provider backlog import walks newest-first, while the CP fit is restricted
    to bests recorded on or before each ride's own date — so at the moment any
    ride is processed during an import, every ride already stored is *newer* than
    it and excluded from its fit. Every ride in a full-history import therefore
    gets fit against a single ride's efforts and freezes that way (issue #77).

    Running this once the import has finished re-fits those rides against the
    now-complete bests table. **This is not a restatement of history**: the "as
    of" restriction still applies per activity, so each ride gets exactly the fit
    it should have had, rather than one taken with data that hadn't arrived yet.
    That is the difference between repairing a snapshot and rewriting one — the
    same line ``backfill_missing_power_best_weights`` draws for bodyweight.

    Scoped by ``since`` (the earliest activity the import touched) so an ordinary
    incremental sync re-fits only the days it actually affected. Note that older
    rides arriving late legitimately change *newer* rides' fits too, which is why
    the range runs forward from the earliest import rather than covering only the
    new rows.

    Walks the history once with a running rank-1 maximum instead of querying per
    activity, keeping this linear rather than quadratic. Returns the number of
    activities whose snapshot changed. The caller commits.
    """
    bests = await cp_fit_bests_ordered(athlete_id, session)

    result = await session.execute(
        select(Activity)
        .where(
            Activity.athlete_id == athlete_id,
            Activity.start_time.is_not(None),
            Activity.start_time >= since,
        )
        .order_by(Activity.start_time)
    )
    activities = list(result.scalars())
    if not activities:
        return 0

    rank1: dict[int, float] = {}
    cursor = 0
    changed = 0

    # Seed with everything at or before the first activity we're re-fitting; the
    # rest is absorbed as the walk advances.
    for activity in activities:
        act_time = _as_utc(activity.start_time)
        while cursor < len(bests):
            best_time = _as_utc(bests[cursor][0])
            if best_time is None or best_time > act_time:
                break
            _, duration_s, power_w = bests[cursor]
            if power_w > rank1.get(duration_s, 0.0):
                rank1[duration_s] = power_w
            cursor += 1

        cp, w_prime, fit_points = fit_cp_wprime(rank1)
        cp_w = round(cp, 1) if cp is not None else None
        w_prime_j = round(w_prime) if w_prime is not None else None

        if (
            activity.cp_w == cp_w
            and activity.w_prime_j == w_prime_j
            and activity.cp_fit_points == fit_points
        ):
            continue

        activity.cp_w = cp_w
        activity.w_prime_j = w_prime_j
        activity.cp_fit_points = fit_points
        changed += 1

        await _rebuild_w_bal(session, activity, cp, w_prime)

    return changed


async def _rebuild_w_bal(
    session: AsyncSession,
    activity: Activity,
    cp: float | None,
    w_prime: float | None,
) -> None:
    """Recompute one activity's ``w_bal`` stream against a changed CP/W' pair.

    Loads the power stream for this activity alone rather than up front for all
    of them — a season of rides is a lot of per-second JSON to hold at once, and
    only the activities whose fit actually moved need it.
    """
    row = await session.execute(
        select(ActivityStream.data).where(
            ActivityStream.activity_id == activity.id,
            ActivityStream.stream_type == "power",
        )
    )
    stored = row.scalar_one_or_none()
    power = [float(v) for v in (stored or [])]

    if not any(v > 0 for v in power) or not _sampling_supports_integration(
        power, activity.duration_s
    ):
        await replace_w_bal_stream(session, activity.id, [])
        return

    await replace_w_bal_stream(
        session, activity.id, [round(v) for v in w_bal_stream(power, cp, w_prime)]
    )
