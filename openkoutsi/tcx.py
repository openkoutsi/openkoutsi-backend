"""TCX activity files → :class:`openkoutsi.workout.Profile`.

Same three-function contract as :mod:`openkoutsi.fit` and :mod:`openkoutsi.gpx`
(see :mod:`openkoutsi.activity_formats`).

TCX sits between the other two in what it preserves. Like GPX it is XML built
around a coordinate track; unlike GPX it is a *training* format, so it states
cumulative distance per track point, carries power and cadence in the standard
activity extension, and — the reason it is worth parsing separately rather than
treating as GPX with extras — it records **laps**. A TCX from an interval
session comes back with the athlete's own splits instead of the arbitrary
auto-split a GPX gets.

Location is handled exactly as in :mod:`openkoutsi.gpx`: coordinates are read
only as a fallback for distance when the file states none, and are never part of
the returned ``Profile``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import NamedTuple

from . import geo, streams, workout
from .activity_formats import ActivityParseError
from .xmlsafe import (
    Fileish,
    XmlSafetyError,
    iter_elements,
    local_name,
    read_bytes,
    root_tag,
    text_float,
)

__all__ = ["ActivityParseError", "extractIntervals", "getStartTime", "summarizeWorkout"]

# TCX puts power and per-point speed in the Garmin activity extension (``TPX``),
# whose namespace prefix varies by writer (``ns3`` from Garmin, ``ns2`` or none
# from others), so these are matched on local name like every other tag here.
_TPX_CHANNELS = {
    "watts": "power",
    "speed": "speed",
    "runcadence": "cadence",
}

# The sport strings TCX actually uses. ``Other`` is left alone deliberately: it
# is what a device writes for anything that is not a run or a ride, and turning
# it into a guess would be worse than passing it through.
_SPORT_ATTR = {
    "biking": "Ride",
    "running": "Run",
}

_MAX_SPEED_INTERVAL_S = 60.0
_MAX_SPEED_MS = 60.0

# An activity name longer than this is a description, or a crafted file
# trying to write a multi-megabyte string into a column every list endpoint
# echoes back.
_MAX_NAME_CHARS = 120


class _Point(NamedTuple):
    time: datetime | None
    lat: float | None
    lon: float | None
    elevation: float | None
    distance: float | None  # cumulative metres, as stated by the device
    channels: dict[str, float]


class _Lap(NamedTuple):
    start_time: datetime | None
    duration_s: float | None
    distance_m: float | None


class _Activity(NamedTuple):
    points: list[_Point]
    laps: list[_Lap]
    sport: str | None
    name: str | None


def _parse_time(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _read_point(elem) -> _Point:
    time = lat = lon = elevation = distance = None
    channels: dict[str, float] = {}

    for child in elem:
        tag = local_name(child.tag)
        if tag == "Time":
            time = _parse_time(child.text)
        elif tag == "Position":
            for coord in child:
                coord_tag = local_name(coord.tag)
                if coord_tag == "LatitudeDegrees":
                    lat = text_float(coord)
                elif coord_tag == "LongitudeDegrees":
                    lon = text_float(coord)
        elif tag == "AltitudeMeters":
            elevation = text_float(child)
        elif tag == "DistanceMeters":
            distance = text_float(child)
        elif tag == "HeartRateBpm":
            for value in child:
                if local_name(value.tag) == "Value":
                    hr = text_float(value)
                    if hr is not None:
                        channels["heartRate"] = hr
        elif tag == "Cadence":
            cadence = text_float(child)
            if cadence is not None:
                channels["cadence"] = cadence
        elif tag == "Extensions":
            for node in child.iter():
                channel = _TPX_CHANNELS.get(local_name(node.tag).lower())
                if channel is None:
                    continue
                value = text_float(node)
                if value is not None:
                    # A bike TCX carries `Cadence` *and* `RunCadence`; the
                    # explicit element is the device's own reading.
                    channels.setdefault(channel, value)

    return _Point(
        time=time,
        lat=lat,
        lon=lon,
        elevation=elevation,
        distance=distance,
        channels=channels,
    )


def _read_lap(elem) -> _Lap:
    duration = distance = None
    for child in elem:
        tag = local_name(child.tag)
        if tag == "TotalTimeSeconds":
            duration = text_float(child)
        elif tag == "DistanceMeters":
            distance = text_float(child)
    return _Lap(
        start_time=_parse_time(elem.get("StartTime")),
        duration_s=duration,
        distance_m=distance,
    )


def _parse(fileish: Fileish) -> _Activity:
    """One pass over the file, collecting track points and laps in document order.

    A TCX may hold several ``<Activity>`` elements. They are concatenated: a
    multi-activity file is something a bulk export produces by accident far more
    often than a user means to record two sessions in one file, and the shared
    clock puts them in order regardless.
    """
    try:
        data = read_bytes(fileish)
    except OSError as exc:
        raise ActivityParseError(f"Could not read the file: {exc}") from exc

    if not data.strip():
        raise ActivityParseError("File is empty")

    points: list[_Point] = []
    laps: list[_Lap] = []
    sport: str | None = None
    name: str | None = None
    # From the head of the file rather than by waiting for the root to close:
    # the root encloses the document, and a wanted element that is open for the
    # whole parse stops `iter_elements` unlinking anything (see its docstring).
    is_tcx = (root_tag(data) or "").lower() == "trainingcenterdatabase"

    try:
        for elem in iter_elements(data, frozenset({"Activity", "Lap", "Trackpoint"})):
            tag = local_name(elem.tag)
            if tag == "Trackpoint":
                points.append(_read_point(elem))
            elif tag == "Lap":
                laps.append(_read_lap(elem))
            elif tag == "Activity":
                if sport is None:
                    sport = (elem.get("Sport") or "").strip() or None
                for child in elem:
                    # Strava writes the ride's title into `Notes`; Garmin leaves
                    # it out. Long notes are a description, not a name.
                    if local_name(child.tag) == "Notes" and name is None:
                        text = (child.text or "").strip()
                        if text and len(text) <= _MAX_NAME_CHARS:
                            name = text
    except XmlSafetyError as exc:
        raise ActivityParseError(str(exc)) from exc

    if not is_tcx and not points:
        raise ActivityParseError(
            "File does not look like TCX (no <TrainingCenterDatabase> element)"
        )
    if not points:
        raise ActivityParseError("TCX file contains no track points")

    return _Activity(points=points, laps=laps, sport=sport, name=name)


def _stated_distance(points: list[_Point]) -> float | None:
    """Total metres from the device's own cumulative distance, if it wrote one.

    Accumulated from the deltas rather than read off the last point: a file
    assembled from several activities, or a device that zeroes its odometer at a
    lap, would otherwise report the last leg alone.
    """
    values = [p.distance for p in points if p.distance is not None]
    if len(values) < 2:
        return None
    total = 0.0
    previous = values[0]
    for value in values[1:]:
        if value >= previous:
            total += value - previous
        else:
            total += value  # counter reset; the new reading is fresh distance
        previous = value
    return total


def _derived_speed(points: list[_Point]) -> dict[int, float]:
    """Speed in m/s by point index, for the points the file did not state one for."""
    out: dict[int, float] = {}
    previous_index: int | None = None
    for index, point in enumerate(points):
        if point.time is None:
            continue
        if previous_index is not None and "speed" not in point.channels:
            previous = points[previous_index]
            dt = (point.time - previous.time).total_seconds()
            step: float | None = None
            if previous.distance is not None and point.distance is not None:
                step = point.distance - previous.distance
            elif None not in (previous.lat, previous.lon, point.lat, point.lon):
                step = geo.haversine_m(previous.lat, previous.lon, point.lat, point.lon)
            if step is not None and 0 <= step <= geo.MAX_STEP_M and 0 < dt <= _MAX_SPEED_INTERVAL_S:
                speed = step / dt
                if speed <= _MAX_SPEED_MS:
                    out[index] = speed
        previous_index = index
    return out


def summarizeWorkout(fileish: Fileish) -> workout.Profile:
    """Parse a TCX activity into a :class:`workout.Profile`.

    Distance comes from the device's own cumulative reading where there is one,
    from the laps' totals otherwise, and only then from the coordinates.
    Duration is the laps' timer time when stated — the same quantity a FIT
    ``session`` reports — falling back to elapsed time.
    """
    activity = _parse(fileish)
    points = activity.points

    distance = _stated_distance(points)
    if distance is None:
        lap_total = sum(lap.distance_m for lap in activity.laps if lap.distance_m)
        distance = lap_total if lap_total else None
    if distance is None:
        distance = geo.track_distance_m(
            [
                (p.lat, p.lon) if p.lat is not None and p.lon is not None else None
                for p in points
            ]
        )

    derived_speed = _derived_speed(points)
    elevation_gain = geo.elevation_gain_m(p.elevation for p in points)

    timestamps: list[datetime] = []
    pending: dict[str, list[tuple[int, float]]] = {
        name: [] for name in ("heartRate", "speed", "power", "cadence", "altitude")
    }

    for index, point in enumerate(points):
        if point.time is None:
            continue
        i = len(timestamps)
        timestamps.append(point.time)

        for channel in ("heartRate", "power", "cadence"):
            value = point.channels.get(channel)
            if value is not None:
                pending[channel].append((i, value))

        speed_ms = point.channels.get("speed")
        if speed_ms is None:
            speed_ms = derived_speed.get(index)
        if speed_ms is not None:
            pending["speed"].append((i, speed_ms * 3.6))  # m/s -> km/h

        if point.elevation is not None:
            pending["altitude"].append((i, point.elevation))

    if not timestamps:
        raise ActivityParseError("TCX file has no track point timestamps")

    offsets, length = streams.second_offsets(timestamps)
    resampled = streams.resample_1hz(
        {
            name: [(offsets[i], value) for i, value in samples if offsets[i] >= 0]
            for name, samples in pending.items()
        },
        length,
    )

    # Timer time when the laps state it, elapsed otherwise — but both bounded by
    # the same cap the stream grid uses, so one absurd `TotalTimeSeconds` or one
    # garbage `<Time>` cannot hand `calculate_load` a duration of centuries. See
    # the note in `gpx.summarizeWorkout`.
    lap_seconds = sum(lap.duration_s for lap in activity.laps if lap.duration_s)
    if lap_seconds > 0:
        duration = min(int(lap_seconds), streams.MAX_STREAM_SECONDS)
    else:
        duration = max(0, length - 1)

    sport = activity.sport
    if sport is not None:
        sport = _SPORT_ATTR.get(sport.lower(), sport)

    return workout.Profile(
        start_time=timestamps[0],
        duration=duration,
        distance=int(round(distance)),
        elevationGain=int(round(elevation_gain)),
        heartRate=resampled["heartRate"],
        speed=resampled["speed"],
        power=resampled["power"],
        cadence=resampled["cadence"],
        altitude=resampled["altitude"],
        sport_type=sport,
        name=activity.name,
    )


def getStartTime(fileish: Fileish) -> datetime | None:
    """The first track point's timestamp, or ``None`` if the file cannot be read.

    Stops at the first timestamp rather than parsing the file — see
    :func:`openkoutsi.gpx.getStartTime` for why that matters to a bulk import.
    """
    try:
        data = read_bytes(fileish)
        # Anchored to `<Trackpoint>` for the same reason GPX is: `<Time>` is a
        # track point's in the activity schema, but a `<Courses>` section in the
        # same file has its own, and matching the bare tag would make which one
        # wins a question about document order.
        for point in iter_elements(data, frozenset({"Trackpoint"})):
            for child in point:
                if local_name(child.tag) == "Time":
                    when = _parse_time(child.text)
                    if when is not None:
                        return when
    except (ActivityParseError, XmlSafetyError, OSError):
        return None
    return None


def extractIntervals(fileish: Fileish) -> list[dict]:
    """The activity's laps, in the shape ``fit.extractIntervals`` returns.

    A lap with no start time or no duration is dropped rather than guessed at;
    if that leaves one lap or none, the caller auto-splits exactly as it does
    for a FIT file with no lap frames.
    """
    try:
        activity = _parse(fileish)
    except (ActivityParseError, XmlSafetyError):
        return []

    intervals = [
        {
            "start_time": lap.start_time,
            "duration_s": float(lap.duration_s),
            "distance_m": float(lap.distance_m) if lap.distance_m is not None else None,
        }
        for lap in activity.laps
        if lap.start_time is not None and lap.duration_s
    ]
    intervals.sort(key=lambda iv: iv["start_time"])
    return intervals
