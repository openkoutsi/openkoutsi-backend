from datetime import datetime, timezone
from typing import cast, Optional

import fitdecode

from . import workout


def extractIntervals(fileish) -> list[dict]:
    """Extract interval (lap) data from a FIT file.

    FIT files use 'lap' frames to record splits. Returns a list of dicts
    with start_time, duration_s, and distance_m — one per lap frame.
    Returns [] if no lap frames are present (caller should auto-split).
    """
    intervals: list[dict] = []
    try:
        with fitdecode.FitReader(fileish) as fr:
            for frame in fr:
                if frame.frame_type != fitdecode.FIT_FRAME_DATA:
                    continue
                frame = cast(fitdecode.records.FitDataMessage, frame)
                if frame.name != "lap":
                    continue
                start_time = frame.get_value("start_time", fallback=None)
                duration_s = frame.get_value("total_timer_time", fallback=None)
                distance_m = frame.get_value("total_distance", fallback=None)
                if start_time is None or duration_s is None:
                    continue
                if isinstance(start_time, datetime) and start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=timezone.utc)
                intervals.append({
                    "start_time": start_time,
                    "duration_s": float(duration_s),
                    "distance_m": float(distance_m) if distance_m is not None else None,
                })
    except Exception:
        pass
    intervals.sort(key=lambda x: x["start_time"])
    return intervals

def getStartTime(fileish) -> Optional[datetime]:
    try:
        with fitdecode.FitReader(fileish) as fr:
            for frame in fr:
                if frame.frame_type != fitdecode.FIT_FRAME_DATA:
                    continue
                frame = cast(fitdecode.records.FitDataMessage, frame)
                
                if frame.name == "record":
                    ts = frame.get_value("timestamp")
                    if ts is not None:
                        if isinstance(ts, datetime) and ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        return ts
    except Exception:
        pass
    return None

def summarizeWorkout(fileish) -> workout.Profile:
    fr = fitdecode.FitReader(fileish)

    first_ts: datetime | None = None
    last_ts = None
    duration_from_session = None
    distance_from_session = 0
    elevation_gain_from_session = 0
    sport_from_file: str | None = None

    heart_rate: list[float] = []
    speed: list[float] = []
    power: list[float] = []
    cadence: list[float] = []
    altitude: list[float] = []

    for frame in fr:
        if frame.frame_type != fitdecode.FIT_FRAME_DATA:
            continue

        frame = cast(fitdecode.records.FitDataMessage, frame)

        if frame.name == "sport":
            s = frame.get_value("sport", fallback=None)
            if s is not None:
                sport_from_file = str(s)

        elif frame.name == "session":
            total_timer = frame.get_value("total_timer_time", fallback=None)
            if total_timer is not None:
                duration_from_session = int(total_timer)

            total_distance = frame.get_value("total_distance", fallback=None)
            if total_distance is not None:
                distance_from_session = int(total_distance)

            total_ascent = frame.get_value("total_ascent", fallback=None)
            if total_ascent is not None:
                elevation_gain_from_session = int(total_ascent)

        elif frame.name == "record":
            ts = frame.get_value("timestamp", fallback=None)
            if ts is not None:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts

            hr = frame.get_value("heart_rate", fallback=None)
            if hr is not None:
                heart_rate.append(float(hr))

            spd = frame.get_value("speed", fallback=None)
            if spd is not None:
                speed.append(float(spd) * 3.6)  # m/s -> km/h

            pwr = frame.get_value("power", fallback=None)
            if pwr is not None:
                power.append(float(pwr))

            cad = frame.get_value("cadence", fallback=None)
            if cad is not None:
                cadence.append(float(cad))

            alt = frame.get_value("altitude", fallback=None)
            if alt is not None:
                altitude.append(float(alt))

    if duration_from_session is not None:
        duration = duration_from_session
    elif first_ts is not None and last_ts is not None:
        if hasattr(last_ts - first_ts, "total_seconds"):
            duration = int((last_ts - first_ts).total_seconds())
        else:
            duration = int(last_ts - first_ts)
    else:
        duration = 0

    return workout.Profile(
        start_time=first_ts or datetime.fromtimestamp(0),
        duration=duration,
        distance=distance_from_session,
        elevationGain=elevation_gain_from_session,
        heartRate=heart_rate,
        speed=speed,
        power=power,
        cadence=cadence,
        altitude=altitude,
        sport_type=sport_from_file,
    )
