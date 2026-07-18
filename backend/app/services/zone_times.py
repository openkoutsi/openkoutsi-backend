"""Per-activity time-in-zone snapshots (issue #27).

Each activity stores its accumulated time-in-zone (power + HR) as a
``zone_times`` snapshot, computed once from its per-second streams using the
athlete's zone definitions in effect at that moment. Once set the snapshot is
frozen, so editing zones later never changes historical activities — only new
ones pick up the new boundaries.
"""
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.user_orm import Activity, ActivityStream, Athlete
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
