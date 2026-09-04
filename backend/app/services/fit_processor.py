import asyncio
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from openkoutsi.activity_formats import parser_for
from openkoutsi.fit import getStartTime
from openkoutsi.categorization import classify_workout
from openkoutsi.workout import Profile
from openkoutsi.streams import to_json_stream
from openkoutsi.fit_processing import (
    resolve_sport_type,
    auto_interval_s,
    build_auto_intervals,
    compute_interval_stats,
)
from openkoutsi.training_math import (
    weighted_power,
    calculate_load,
    compute_power_bests,
    compute_distance_bests,
    compute_torque_stream,
    variability_index,
)
from backend.app.models.user_orm import (
    Activity,
    ActivityDistanceBest,
    ActivityInterval,
    ActivityPowerBest,
    ActivityStream,
    Athlete,
)
from backend.app.services.aerobic_metrics import apply_aerobic_metrics
from backend.app.services.commute import evaluate_activity
from backend.app.services.garage import assign_bike
from backend.app.services.weight import effective_weight_for, load_weight_log, w_per_kg
from backend.app.services.zone_times import compute_zone_times


def read_fit_start_time(path: str) -> Optional[datetime]:
    """
    Extract just the start timestamp from a FIT file without full processing.
    Reads only until the first data record, so it's fast even for large files.
    Returns a UTC-aware datetime, or None if the file contains no timestamps.
    """
    return getStartTime(path)


def read_activity_start_time(path: str, fmt: str = "fit") -> Optional[datetime]:
    """The start timestamp of an activity file in any supported format.

    Every parser stops at the first timestamped record, so this stays cheap
    enough to run over every file in a bulk import before deciding which ones
    are duplicates of each other.
    """
    return parser_for(fmt).getStartTime(path)


def parse_activity_file(path: str, fmt: str = "fit") -> tuple[Profile, list[dict]]:
    """Parse a file into its profile and its device-recorded laps.

    Split out from :func:`process_activity_file` because parsing is the
    expensive, purely synchronous half: a bulk import runs this in a worker
    thread so that decoding nine hundred files does not block the event loop,
    then hands the result back to be written. Raises
    ``openkoutsi.activity_formats.ActivityParseError`` for a file that cannot be
    read, carrying a reason fit to show an athlete.
    """
    parser = parser_for(fmt)
    return parser.summarizeWorkout(path), parser.extractIntervals(path)


async def process_fit_file(
    path: str,
    athlete: Athlete,
    activity: Activity,
    session: AsyncSession,
) -> Activity:
    """Populate an activity from a FIT file. See :func:`process_activity_file`."""
    return await process_activity_file(path, athlete, activity, session)


async def process_activity_file(
    path: str,
    athlete: Athlete,
    activity: Activity,
    session: AsyncSession,
    *,
    fmt: str = "fit",
    parsed: Optional[tuple[Profile, list[dict]]] = None,
) -> Activity:
    """Populate an activity — metrics, streams, bests, intervals — from a file.

    Format-agnostic since issue #36: FIT, GPX and TCX all arrive here as a
    ``Profile``, and everything below this line is the same work regardless of
    which one it came from. What differs is only what the file *had* — a GPX
    with no power produces no weighted power, no power bests and no power zone
    times, and that is a complete import of that file rather than a failure.

    Pass ``parsed`` when the caller has already parsed the file (in a thread,
    for a bulk import) to avoid doing it twice. Otherwise the parse happens here
    in a worker thread: it is pure-Python iteration over the whole file (11.2 s
    for a 4.8 MB ride, under a tenth of the upload limit), and this function is
    awaited from a ``BackgroundTasks`` callback that Starlette runs on the event
    loop rather than in a threadpool — left inline it stalls every other request
    for the length of the parse (issue #101 §2.2, issue #102 F-05).
    """
    if parsed is not None:
        profile, raw_intervals = parsed
    else:
        profile, raw_intervals = await asyncio.to_thread(
            parse_activity_file, path, fmt
        )

    wp = weighted_power(profile.power) if profile.power else None
    load, intensity = calculate_load(
        profile.duration,
        wp,
        profile.avgHeartRate if profile.heartRate else None,
        athlete.ftp,
        athlete.max_hr,
    )

    # GPX and TCX carry the ride's own title; FIT almost never does.
    activity.name = activity.name or profile.name or "Uploaded Activity"
    activity.sport_type = activity.sport_type or resolve_sport_type(profile.sport_type)
    activity.start_time = profile.start_time
    activity.duration_s = profile.duration
    activity.distance_m = float(profile.distance)
    activity.elevation_m = float(profile.elevationGain)
    activity.avg_power = profile.avgPower if profile.power else None
    activity.weighted_power = wp
    activity.avg_hr = profile.avgHeartRate if profile.heartRate else None
    activity.max_hr = profile.peakHR if profile.heartRate else None
    activity.avg_speed_ms = (profile.avgSpeed / 3.6) if profile.speed else None
    activity.avg_cadence = float(profile.avgCadence) if profile.cadence else None
    activity.load = load
    activity.intensity = intensity
    activity.status = "processed"

    # Categorize before deriving the aerobic metrics: the decoupling gate uses
    # the category (and the same variability index) to spot interval sessions,
    # where a power:HR drift number would describe the intervals rather than the
    # athlete's durability.
    vi = variability_index(wp, activity.avg_power)
    category = classify_workout(intensity, vi)
    activity.workout_category = category.value if category else None

    # Commute detection reads distance, duration and the local clock, all of
    # which are set above, so it goes after them and before the commit (issue
    # #63). It writes a *suggestion*, never the label — see
    # `services.commute` for why those have to stay apart.
    await evaluate_activity(session, athlete, activity)

    # And which bike it was ridden on (issue #64). Unlike the commute
    # suggestion this one is *applied*, but only over an empty or previously
    # automatic assignment — never over the athlete's own choice.
    await assign_bike(session, athlete, activity)

    # Every channel is on the one 1 Hz clock the parser resampled onto, gaps as
    # None (issue #76), and ``to_json_stream`` is what keeps a NaN from ever
    # reaching the JSON column.
    power_data = to_json_stream(profile.power)
    cadence_data = to_json_stream(profile.cadence)
    stream_map: dict[str, list[float | None]] = {
        "power": power_data,
        "heartrate": to_json_stream(profile.heartRate),
        "cadence": cadence_data,
        "speed": to_json_stream(
            [None if v is None else v / 3.6 for v in profile.speed]  # km/h -> m/s
        ),
        "altitude": to_json_stream(profile.altitude),
        "torque": compute_torque_stream(power_data, cadence_data),
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

    # Freeze time-in-zone using the athlete's zones as they are right now (issue
    # #27). Editing zones later won't rewrite this activity's snapshot.
    activity.zone_times = compute_zone_times(
        stream_map, athlete.hr_zones, athlete.power_zones
    )

    if power_data:
        weight_log = await load_weight_log(athlete.id, session)
        act_date = activity.start_time.date() if activity.start_time else None
        weight = effective_weight_for(weight_log, act_date)
        bests = compute_power_bests(power_data)
        for duration_s, power_w in bests.items():
            session.add(
                ActivityPowerBest(
                    activity_id=activity.id,
                    athlete_id=athlete.id,
                    duration_s=duration_s,
                    power_w=power_w,
                    activity_start_time=activity.start_time,
                    weight_kg=weight,
                    w_per_kg=w_per_kg(power_w, weight),
                )
            )

    # Aerobic decoupling, the CP/W' snapshot and the W' balance stream (issue
    # #37). Runs after the power bests are added so the CP fit — which is
    # restricted to bests as of this activity's date — includes this ride's own
    # efforts.
    w_bal_data = await apply_aerobic_metrics(activity, athlete, stream_map, session)
    if w_bal_data:
        stream_map["w_bal"] = w_bal_data
        session.add(
            ActivityStream(
                id=str(uuid.uuid4()),
                activity_id=activity.id,
                stream_type="w_bal",
                data=w_bal_data,
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

    await session.commit()
    await session.refresh(activity)
    return activity
