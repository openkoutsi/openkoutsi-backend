"""Per-activity time-in-zone snapshots (issue #27).

Each activity stores its accumulated time-in-zone (power + HR) as a
``zone_times`` snapshot, computed once from its per-second streams using the
athlete's zone definitions in effect at that moment. Once set the snapshot is
frozen, so editing zones later never changes historical activities — only new
ones pick up the new boundaries.
"""
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.user_orm import Activity, ActivityStream, Athlete
from openkoutsi.sport_matching import CYCLING_SPORT_TYPES
from openkoutsi.zones import time_in_zones


def compute_zone_times(
    streams: dict[str, list],
    hr_zones: Sequence[dict] | None,
    power_zones: Sequence[dict] | None,
) -> dict | None:
    """Build a ``{"hr": {...}, "power": {...}}`` snapshot from streams + zones.

    Returns ``None`` when nothing can be computed (no configured zones, or no
    matching stream), so callers can leave ``zone_times`` unset rather than
    persisting an empty snapshot.
    """
    result: dict[str, dict[str, int]] = {}
    if hr_zones and streams.get("heartrate"):
        result["hr"] = time_in_zones(streams["heartrate"], hr_zones)
    if power_zones and streams.get("power"):
        result["power"] = time_in_zones(streams["power"], power_zones)
    return result or None


async def ensure_zone_times(
    athlete: Athlete,
    session: AsyncSession,
    activities: Sequence[Activity],
) -> int:
    """Backfill missing ``zone_times`` for the given activities.

    Only touches activities whose snapshot is unset — already-frozen snapshots
    are left alone. Uses the athlete's *current* zones (the best available
    reference for rides recorded before snapshots existed). The caller is
    responsible for committing. Returns the number of activities updated.
    """
    if not athlete.hr_zones and not athlete.power_zones:
        return 0

    pending = [a for a in activities if a.zone_times is None]
    if not pending:
        return 0

    ids = [a.id for a in pending]
    streams_result = await session.execute(
        select(ActivityStream).where(ActivityStream.activity_id.in_(ids))
    )
    streams_by_activity: dict[str, dict[str, list]] = {}
    for s in streams_result.scalars():
        streams_by_activity.setdefault(s.activity_id, {})[s.stream_type] = s.data

    updated = 0
    for activity in pending:
        streams = streams_by_activity.get(activity.id)
        if not streams:
            continue
        zone_times = compute_zone_times(streams, athlete.hr_zones, athlete.power_zones)
        if zone_times is not None:
            activity.zone_times = zone_times
            updated += 1
    return updated


async def weekly_zone_buckets(
    athlete: Athlete,
    session: AsyncSession,
    *,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> list[dict]:
    """Accumulated time-in-zone per Monday-based week over a period.

    Sums each processed cycling activity's frozen ``zone_times`` snapshot into
    weekly buckets, shaped as ``{"week_start": date, "hr": {...}, "power": {...}}``
    and ordered oldest first. Legacy activities with no snapshot are backfilled
    (using current zones) and frozen on the way through, mirroring the fitness
    catch-up flow — so this commits when it wrote something.

    Lives here rather than in the route because two callers need exactly this
    answer: ``GET /api/metrics/zones/weekly`` and the ``get_zone_totals`` MCP
    tool (issue #42). Two implementations of "which week does this ride belong
    to" would eventually disagree about a Sunday night ride.
    """
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
        {
            "week_start": week_start,
            "hr": data.get("hr", {}),
            "power": data.get("power", {}),
        }
        for week_start, data in sorted(buckets.items())
    ]
