import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from openkoutsi.fit import summarizeWorkout, getStartTime, extractIntervals
from openkoutsi.categorization import classify_workout
from openkoutsi.fit_processing import (
    resolve_sport_type,
    auto_interval_s,
    build_auto_intervals,
    compute_interval_stats,
)
from openkoutsi.training_math import (
    normalized_power,
    calculate_tss,
    compute_power_bests,
    compute_distance_bests,
)
from backend.app.models.team_orm import (
    Activity,
    ActivityDistanceBest,
    ActivityInterval,
    ActivityPowerBest,
    ActivityStream,
    Athlete,
)


def read_fit_start_time(path: str) -> Optional[datetime]:
    """
    Extract just the start timestamp from a FIT file without full processing.
    Reads only until the first data record, so it's fast even for large files.
    Returns a UTC-aware datetime, or None if the file contains no timestamps.
    """
    return getStartTime(path)


async def process_fit_file(
    path: str,
    athlete: Athlete,
    activity: Activity,
    session: AsyncSession,
) -> Activity:
    profile = summarizeWorkout(path)

    np = normalized_power(profile.power) if profile.power else None
    tss, intensity_factor = calculate_tss(
        profile.duration,
        np,
        profile.avgHeartRate if profile.heartRate else None,
        athlete.ftp,
        athlete.max_hr,
    )

    activity.name = activity.name or "Uploaded Activity"
    activity.sport_type = activity.sport_type or resolve_sport_type(profile.sport_type)
    activity.start_time = profile.start_time
    activity.duration_s = profile.duration
    activity.distance_m = float(profile.distance)
    activity.elevation_m = float(profile.elevationGain)
    activity.avg_power = profile.avgPower if profile.power else None
    activity.normalized_power = np
    activity.avg_hr = profile.avgHeartRate if profile.heartRate else None
    activity.max_hr = profile.peakHR if profile.heartRate else None
    activity.avg_speed_ms = (profile.avgSpeed / 3.6) if profile.speed else None
    activity.avg_cadence = float(profile.avgCadence) if profile.cadence else None
    activity.tss = tss
    activity.intensity_factor = intensity_factor
    activity.status = "processed"

    power_data = [float(v) for v in profile.power]
    stream_map = {
        "power": power_data,
        "heartrate": [float(v) for v in profile.heartRate],
        "cadence": [float(v) for v in profile.cadence],
        "speed": [v / 3.6 for v in profile.speed],  # km/h -> m/s
        "altitude": [float(v) for v in profile.altitude],
    }
    for stream_type, data in stream_map.items():
        if data:
            session.add(
                ActivityStream(
                    id=str(uuid.uuid4()),
                    activity_id=activity.id,
                    stream_type=stream_type,
                    data=data,
                )
            )

    if power_data:
        bests = compute_power_bests(power_data)
        for duration_s, power_w in bests.items():
            session.add(
                ActivityPowerBest(
                    activity_id=activity.id,
                    athlete_id=athlete.id,
                    duration_s=duration_s,
                    power_w=power_w,
                    activity_start_time=activity.start_time,
                )
            )

    speed_data_ms = stream_map["speed"]
    if speed_data_ms:
        dbests = compute_distance_bests(speed_data_ms)
        for distance_m, time_s in dbests.items():
            session.add(
                ActivityDistanceBest(
                    activity_id=activity.id,
                    athlete_id=athlete.id,
                    distance_m=distance_m,
                    time_s=time_s,
                    activity_start_time=activity.start_time,
                )
            )

    raw_intervals = extractIntervals(path)
    is_auto = len(raw_intervals) <= 1
    if is_auto:
        stream_length = max(
            (len(v) for v in stream_map.values() if v), default=profile.duration
        )
        actual_duration = max(profile.duration, stream_length)
        interval_s = auto_interval_s(actual_duration)
        raw_intervals = build_auto_intervals(profile.start_time, actual_duration, interval_s)

    intervals = compute_interval_stats(raw_intervals, profile.start_time, stream_map, is_auto)
    for iv in intervals:
        session.add(ActivityInterval(id=str(uuid.uuid4()), activity_id=activity.id, **iv))

    vi = (np / activity.avg_power) if (np and activity.avg_power) else None
    category = classify_workout(intensity_factor, vi)
    activity.workout_category = category.value if category else None

    await session.commit()
    await session.refresh(activity)
    return activity
