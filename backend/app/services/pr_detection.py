from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.team_orm import Activity, ActivityDistanceBest, ActivityPowerBest
from openkoutsi.training_math import DISTANCE_BEST_DISTANCES, POWER_BEST_DURATIONS

_WINDOWS: dict[str, timedelta | None] = {
    "all_time": None,
    "12mo": timedelta(days=365),
    "6mo": timedelta(days=182),
    "3mo": timedelta(days=91),
}

_TIERS = {1: "gold", 2: "silver", 3: "bronze"}


def _is_virtual(sport_type: str | None) -> bool:
    return (sport_type or "").startswith("Virtual")


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _rank_in_window(
    rows: list,
    *,
    target_id: str,
    cutoff: datetime | None,
    key_attr: str,
    ascending: bool,
) -> str | None:
    """
    Within a time window, rank rows by key_attr (ascending=lower-is-better for time,
    descending for power). Ties broken by earlier activity_start_time.
    Returns the tier string if target_id is in the top 3, else None.
    """
    filtered = [
        r for r in rows
        if r.activity_start_time is not None
        and (cutoff is None or _as_utc(r.activity_start_time) >= cutoff)
    ]
    filtered.sort(key=lambda r: (
        getattr(r, key_attr) if ascending else -getattr(r, key_attr),
        _as_utc(r.activity_start_time),
    ))
    for rank, r in enumerate(filtered[:3], start=1):
        if r.activity_id == target_id:
            return _TIERS[rank]
    return None


async def detect_pr_badges(
    athlete_id: str,
    activity_id: str,
    activity_start_time: datetime | None,
    sport_type: str | None,
    session: AsyncSession,
) -> tuple[dict[int, dict[str, str]], dict[int, dict[str, str]]]:
    """
    Compute PR badge tiers for a given activity across 4 time windows.

    Returns (power_pr_badges, distance_pr_badges) where each is a sparse dict:
      {key: {window: tier}}  e.g. {300: {"all_time": "gold", "12mo": "gold"}}

    Power PRs compare all activities regardless of sport type.
    Distance PRs compare only within the same virtual/real category.
    """
    if activity_start_time is None:
        return {}, {}

    ref = _as_utc(activity_start_time)

    # ── Power bests ──────────────────────────────────────────────────────────
    power_result = await session.execute(
        select(ActivityPowerBest).where(ActivityPowerBest.athlete_id == athlete_id)
    )
    by_duration: dict[int, list[ActivityPowerBest]] = defaultdict(list)
    for r in power_result.scalars():
        by_duration[r.duration_s].append(r)

    power_badges: dict[int, dict[str, str]] = {}
    for duration_s in POWER_BEST_DURATIONS:
        rows = by_duration.get(duration_s, [])
        badges: dict[str, str] = {}
        for window, delta in _WINDOWS.items():
            cutoff = (ref - delta) if delta is not None else None
            tier = _rank_in_window(rows, target_id=activity_id, cutoff=cutoff, key_attr="power_w", ascending=False)
            if tier:
                badges[window] = tier
        if badges:
            power_badges[duration_s] = badges

    # ── Distance bests ───────────────────────────────────────────────────────
    dist_result = await session.execute(
        select(ActivityDistanceBest, Activity.sport_type)
        .join(Activity, Activity.id == ActivityDistanceBest.activity_id)
        .where(ActivityDistanceBest.athlete_id == athlete_id)
    )

    target_virtual = _is_virtual(sport_type)

    class _Row:
        __slots__ = ("activity_id", "distance_m", "time_s", "activity_start_time")

        def __init__(self, best: ActivityDistanceBest) -> None:
            self.activity_id = best.activity_id
            self.distance_m = best.distance_m
            self.time_s = best.time_s
            self.activity_start_time = best.activity_start_time

    by_distance: dict[int, list[_Row]] = defaultdict(list)
    for best, row_sport_type in dist_result.all():
        if _is_virtual(row_sport_type) == target_virtual:
            by_distance[best.distance_m].append(_Row(best))

    distance_badges: dict[int, dict[str, str]] = {}
    for distance_m in DISTANCE_BEST_DISTANCES:
        rows = by_distance.get(distance_m, [])
        badges = {}
        for window, delta in _WINDOWS.items():
            cutoff = (ref - delta) if delta is not None else None
            tier = _rank_in_window(rows, target_id=activity_id, cutoff=cutoff, key_attr="time_s", ascending=True)
            if tier:
                badges[window] = tier
        if badges:
            distance_badges[distance_m] = badges

    return power_badges, distance_badges
