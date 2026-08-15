"""Pure helpers for FIT activity processing — no database dependencies."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Sequence

from . import streams
from .sport_matching import canonical_sport_type


_FIT_SPORT_MAP = {
    "running": "Run",
    "cycling": "Ride",
    "training": "WeightTraining",
    "swimming": "Swim",
    "walking": "Walk",
    "hiking": "Hike",
    # TCX names cycling this way in its ``Sport`` attribute; GPX writers use the
    # spellings above or a canonical Strava name, which the lookup below covers.
    "biking": "Ride",
}


def resolve_sport_type(fit_sport: str | None) -> str:
    """Normalise a raw sport string from an activity file to a Strava-style name.

    Named for the FIT files it was written against, but every format's parser
    goes through it — a GPX ``<type>``, a TCX ``Sport`` attribute and a FIT
    ``sport`` message all arrive here, and all three have their own vocabulary.
    """
    if fit_sport is None:
        return "Cycling"
    mapped = _FIT_SPORT_MAP.get(fit_sport.lower())
    if mapped:
        return mapped
    # A file that already names a sport openkoutsi knows keeps that name rather
    # than being title-cased into a near-miss ("virtualride" → "Virtualride",
    # which matches nothing).
    canonical = canonical_sport_type(fit_sport)
    if canonical:
        return canonical
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


def mean_nonzero(values: Sequence[float | None]) -> Optional[float]:
    """Mean of the positive samples, ignoring the zeros a paused sensor records.

    Gaps drop out for free: ``nan > 0`` is False, so a second the sensor never
    reported is excluded on the same footing as a second it reported zero.
    """
    arr = streams.as_array(values)
    nonzero = arr[arr > 0]
    return float(nonzero.mean()) if nonzero.size else None


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
    stream_map:   dict of stream_type → per-second float list, gaps as None
    is_auto:      whether these are auto-generated (vs. device-recorded) intervals

    The slicing below indexes streams by second offset, which has always assumed
    index == second and, since issue #76, actually gets it: before that a
    dropout shifted every later sample earlier, so a lap's window drifted off
    the seconds it was supposed to cover.
    """
    if activity_start.tzinfo is not None:
        activity_start = activity_start.replace(tzinfo=None)

    # Converted once for the whole activity rather than per interval per stream.
    arrays = {
        key: streams.as_array(data)
        for key, data in stream_map.items()
        if data is not None and len(data)
    }

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
            data = arrays.get(key)
            if data is None:
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
