"""Pure helpers for FIT activity processing — no database dependencies."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional


_FIT_SPORT_MAP = {
    "running": "Run",
    "cycling": "Ride",
    "training": "WeightTraining",
    "swimming": "Swim",
    "walking": "Walk",
    "hiking": "Hike",
}


def resolve_sport_type(fit_sport: str | None) -> str:
    """Normalise a raw fitdecode sport string to a Strava-style name."""
    if fit_sport is None:
        return "Cycling"
    mapped = _FIT_SPORT_MAP.get(fit_sport.lower())
    if mapped:
        return mapped
    return fit_sport.title()


def auto_interval_s(duration_s: int) -> int:
    """Choose auto-split interval length based on total activity duration."""
    minutes = duration_s / 60
    if minutes <= 45:
        return 5 * 60
    elif minutes <= 90:
        return 10 * 60
    else:
        return 15 * 60


def build_auto_intervals(activity_start: datetime, duration_s: int, interval_s: int) -> list[dict]:
    """Produce a list of time-based interval dicts covering the full activity."""
    intervals = []
    offset = 0
    while offset < duration_s:
        length = min(interval_s, duration_s - offset)
        intervals.append({
            "start_time": activity_start + timedelta(seconds=offset),
            "duration_s": float(length),
            "distance_m": None,
        })
        offset += interval_s
    return intervals


def mean_nonzero(values: list[float]) -> Optional[float]:
    nonzero = [v for v in values if v > 0]
    return sum(nonzero) / len(nonzero) if nonzero else None


def compute_interval_stats(
    raw: list[dict],
    activity_start: datetime,
    stream_map: dict[str, list[float]],
    is_auto: bool,
) -> list[dict]:
    """
    Compute per-interval averages from stream data.

    raw:          list of {start_time, duration_s, distance_m}
    activity_start: overall activity start (naive or tz-aware)
    stream_map:   dict of stream_type → per-second float list
    is_auto:      whether these are auto-generated (vs. device-recorded) intervals
    """
    if activity_start.tzinfo is not None:
        activity_start = activity_start.replace(tzinfo=None)

    result = []
    for i, iv in enumerate(raw):
        iv_start = iv["start_time"]
        if isinstance(iv_start, datetime) and iv_start.tzinfo is not None:
            iv_start = iv_start.replace(tzinfo=None)
        start_offset_s = int(round((iv_start - activity_start).total_seconds()))
        duration_s = int(round(iv["duration_s"]))
        start_offset_s = max(0, start_offset_s)
        end = start_offset_s + duration_s

        def _slice_mean(key: str) -> Optional[float]:
            data = stream_map.get(key, [])
            if not data:
                return None
            return mean_nonzero(data[start_offset_s:end])

        result.append({
            "interval_number": i + 1,
            "start_offset_s": start_offset_s,
            "duration_s": duration_s,
            "distance_m": iv.get("distance_m"),
            "avg_hr": _slice_mean("heartrate"),
            "avg_power": _slice_mean("power"),
            "avg_speed_ms": _slice_mean("speed"),
            "avg_cadence": _slice_mean("cadence"),
            "is_auto_split": is_auto,
        })
    return result
