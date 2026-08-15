"""GPX activity files → :class:`openkoutsi.workout.Profile`.

Same contract as :mod:`openkoutsi.fit`: ``summarizeWorkout``, ``getStartTime``
and ``extractIntervals``, so everything downstream of a parsed activity — Load,
weighted power, zone snapshots, power bests, torque, interval extraction — works
on a GPX file without knowing one exists.

Two things a GPX is not:

* **It is not a power file.** Most GPX in the wild is GPS-and-heart-rate, and a
  fair amount is GPS only. ``calculate_load`` already falls back to heart rate,
  so such an activity gets a Load; it simply has no weighted power, no power
  bests and no power zone times. That is the file being honest, not an error.
* **It does not carry the ride's own summary.** A FIT file states its distance,
  ascent and timer time in a ``session`` message. GPX states coordinates and
  nothing else, so distance and elevation gain here are *derived* — see
  :mod:`openkoutsi.geo` — and duration is elapsed time between the first and
  last point rather than moving time.

Location never leaves this module through :func:`summarizeWorkout`. The
coordinates are read, turned into two scalars, and dropped; the ``Profile`` has
no field that could carry them, so nothing downstream can persist what it never
receives. :func:`extract_route` is the deliberate, separate exception — see its
docstring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import NamedTuple

from . import geo, streams, workout
from .activity_formats import ActivityParseError
from .xmlsafe import Fileish, XmlSafetyError, iter_elements, local_name, read_bytes

__all__ = [
    "ActivityParseError",
    "Route",
    "RoutePoint",
    "extractIntervals",
    "extract_route",
    "getStartTime",
    "summarizeWorkout",
]

# Extension element names, lowercased, mapped onto stream channels. Every vendor
# spells these differently — Garmin's TrackPointExtension writes ``hr``/``cad``,
# Strava writes a bare ``power``, the power extension writes ``PowerInWatts`` —
# and they all appear under ``<extensions>`` at some depth, so the parser walks
# the subtree and matches on local name rather than expecting a schema.
_EXTENSION_CHANNELS = {
    "hr": "heartRate",
    "heartrate": "heartRate",
    "heart_rate": "heartRate",
    "cad": "cadence",
    "cadence": "cadence",
    "runcadence": "cadence",
    "run_cadence": "cadence",
    "power": "power",
    "powerinwatts": "power",
    "watts": "power",
    "speed": "speed",  # metres per second, converted below
}

# A gap longer than this between consecutive points is a stopped device, not a
# slow athlete: deriving a speed across it would describe the café rather than
# the ride.
_MAX_SPEED_INTERVAL_S = 60.0
# 216 km/h. Above this the pair of fixes is wrong, not fast.
_MAX_SPEED_MS = 60.0


class _Point(NamedTuple):
    """One track point as read, before location is discarded."""

    time: datetime | None
    lat: float | None
    lon: float | None
    elevation: float | None
    segment: int
    channels: dict[str, float]


@dataclass(frozen=True)
class RoutePoint:
    """A single point of a route, with its coordinates intact."""

    latitude: float
    longitude: float
    elevation_m: float | None = None
    offset_s: int | None = None


@dataclass
class Route:
    """A GPX track's geometry — **the one openkoutsi structure that holds location.**

    Produced only by :func:`extract_route`, which a caller has to ask for by
    name. It exists for the route-analysis work that needs the shape of a ride
    (gradients, climbs, matching one ride's course against another's), and it is
    meant to be computed, used, and thrown away inside a single request.

    It must not be written to the database. openkoutsi's stated position is that
    it stores no location data, ``ActivityStream`` has no channel for it, and the
    ingestion path deliberately goes through :func:`summarizeWorkout`, which
    cannot return one of these. If a future feature needs route data to outlive a
    request, that is a product decision about the privacy promise — and a
    migration, a consent question and a retention rule — not something to reach
    for because this type happens to be available.
    """

    points: list[RoutePoint] = field(default_factory=list)
    start_time: datetime | None = None
    name: str | None = None
    sport_type: str | None = None
    distance_m: float = 0.0
    elevation_gain_m: float = 0.0

    def bounds(self) -> tuple[float, float, float, float] | None:
        """``(min_lat, min_lon, max_lat, max_lon)``, or ``None`` when empty."""
        if not self.points:
            return None
        lats = [p.latitude for p in self.points]
        lons = [p.longitude for p in self.points]
        return (min(lats), min(lons), max(lats), max(lons))


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


def _read_channels(trkpt) -> dict[str, float]:
    """Sensor values hanging off one track point, by stream channel."""
    found: dict[str, float] = {}
    for child in trkpt.iter():
        name = local_name(child.tag).lower()
        channel = _EXTENSION_CHANNELS.get(name)
        if channel is None or child.text is None:
            continue
        try:
            value = float(child.text.strip())
        except (TypeError, ValueError):
            continue
        # First writer wins: a file carrying both `<power>` and
        # `<gpxpx:PowerInWatts>` is stating the same number twice, and if it
        # isn't, the outer (device-native) one came first.
        found.setdefault(channel, value)
    return found


class _Track(NamedTuple):
    points: list[_Point]
    name: str | None
    sport_type: str | None


def _parse(fileish: Fileish) -> _Track:
    """One pass over the file, collecting every track point in document order.

    Multiple ``<trk>`` and ``<trkseg>`` elements are concatenated; the segment
    index rides along on each point so distance and speed are never derived
    across a segment boundary, which is exactly where a device was paused.
    """
    try:
        data = read_bytes(fileish)
    except OSError as exc:
        raise ActivityParseError(f"Could not read the file: {exc}") from exc

    if not data.strip():
        raise ActivityParseError("File is empty")

    points: list[_Point] = []
    name: str | None = None
    sport_type: str | None = None
    segment = 0
    seen_gpx_root = False

    try:
        for elem in iter_elements(data, frozenset({"gpx", "trk", "trkseg", "trkpt"})):
            tag = local_name(elem.tag)
            if tag == "gpx":
                seen_gpx_root = True
                continue
            if tag == "trkseg":
                # Elements arrive on their *end* event, so a segment closes
                # after every point inside it: bumping the counter here leaves
                # each point carrying the index of the segment it was in.
                segment += 1
                continue
            if tag == "trk":
                # Read from the track rather than from `<metadata>`: Strava and
                # Garmin both put the activity's own name here, and the metadata
                # name is as often the exporting tool's.
                for child in elem:
                    child_tag = local_name(child.tag)
                    text = (child.text or "").strip()
                    if not text:
                        continue
                    if child_tag == "name" and name is None:
                        name = text
                    elif child_tag == "type" and sport_type is None:
                        sport_type = text
                continue

            lat = elem.get("lat")
            lon = elem.get("lon")
            time_elem = elevation = None
            for child in elem:
                child_tag = local_name(child.tag)
                if child_tag == "time":
                    time_elem = child.text
                elif child_tag == "ele":
                    elevation = child.text

            try:
                latitude = float(lat) if lat is not None else None
                longitude = float(lon) if lon is not None else None
            except ValueError:
                latitude = longitude = None
            try:
                altitude = float(elevation) if elevation is not None else None
            except (TypeError, ValueError):
                altitude = None

            points.append(
                _Point(
                    time=_parse_time(time_elem),
                    lat=latitude,
                    lon=longitude,
                    elevation=altitude,
                    segment=segment,
                    channels=_read_channels(elem),
                )
            )
    except XmlSafetyError as exc:
        raise ActivityParseError(str(exc)) from exc

    if not seen_gpx_root and not points:
        raise ActivityParseError("File does not look like GPX (no <gpx> element)")
    if not points:
        raise ActivityParseError("GPX file contains no track points")

    return _Track(points=points, name=name, sport_type=sport_type)


def _has_fix(point: _Point) -> bool:
    if point.lat is None or point.lon is None:
        return False
    # (0, 0) is what a receiver writes before it has locked on.
    if point.lat == 0.0 and point.lon == 0.0:
        return False
    return -90.0 <= point.lat <= 90.0 and -180.0 <= point.lon <= 180.0


def _derive(points: list[_Point]) -> tuple[float, dict[int, float]]:
    """Total distance in metres, and derived speed (m/s) keyed by point index.

    Speed is derived only where the file did not state it. Neither figure is
    computed across a ``<trkseg>`` boundary: a new segment means the device was
    paused, and the metres between the last fix before the pause and the first
    after it were not travelled at any speed worth recording.
    """
    distance = 0.0
    derived_speed: dict[int, float] = {}
    previous_index: int | None = None

    for index, point in enumerate(points):
        if not _has_fix(point):
            continue
        if previous_index is not None:
            previous = points[previous_index]
            if previous.segment == point.segment:
                step = geo.haversine_m(previous.lat, previous.lon, point.lat, point.lon)
                if step <= geo.MAX_STEP_M:
                    distance += step
                    if (
                        previous.time is not None
                        and point.time is not None
                        and "speed" not in point.channels
                    ):
                        dt = (point.time - previous.time).total_seconds()
                        if 0 < dt <= _MAX_SPEED_INTERVAL_S:
                            speed = step / dt
                            if speed <= _MAX_SPEED_MS:
                                derived_speed[index] = speed
        previous_index = index

    return distance, derived_speed


def _normalise_sport(raw: str | None) -> str | None:
    """Drop the sport strings that carry no information.

    Strava's older GPX exports write the numeric activity-type id in ``<type>``,
    which would otherwise become an activity called "9".
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text or text.isdigit():
        return None
    return text


def summarizeWorkout(fileish: Fileish) -> workout.Profile:
    """Parse a GPX activity into a :class:`workout.Profile`.

    Streams come back on the shared 1 Hz clock described in
    :mod:`openkoutsi.streams`: index ``i`` is second ``i`` from the first point
    that carried a timestamp, gaps as ``None``. **Coordinates are consumed here
    and are not part of the result.**
    """
    track = _parse(fileish)
    points = track.points

    distance, derived_speed = _derive(points)
    elevation_gain = geo.elevation_gain_m(p.elevation for p in points)

    timestamps: list[datetime] = []
    pending: dict[str, list[tuple[int, float]]] = {
        name: [] for name in ("heartRate", "speed", "power", "cadence", "altitude")
    }

    for index, point in enumerate(points):
        if point.time is None:
            # No place on the clock. A GPX with no times at all is a route, not
            # an activity, and is rejected below; a single point missing one is
            # simply not sampled, exactly as `fit.summarizeWorkout` treats a
            # record with no timestamp.
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
        raise ActivityParseError(
            "GPX file has no timestamps — it is a route or waypoint file, not a recorded activity"
        )

    offsets, length = streams.second_offsets(timestamps)
    resampled = streams.resample_1hz(
        {
            name: [(offsets[i], value) for i, value in samples if offsets[i] >= 0]
            for name, samples in pending.items()
        },
        length,
    )

    first = timestamps[0]
    last = max(timestamps)
    duration = max(0, int((last - first).total_seconds()))

    return workout.Profile(
        start_time=first,
        duration=duration,
        distance=int(round(distance)),
        elevationGain=int(round(elevation_gain)),
        heartRate=resampled["heartRate"],
        speed=resampled["speed"],
        power=resampled["power"],
        cadence=resampled["cadence"],
        altitude=resampled["altitude"],
        sport_type=_normalise_sport(track.sport_type),
        name=track.name,
    )


def getStartTime(fileish: Fileish) -> datetime | None:
    """The first track point's timestamp, or ``None``.

    Used for the duplicate window before an activity row is created, so it
    tolerates a file it cannot parse rather than raising — the caller learns the
    file is unusable when it is actually parsed.
    """
    try:
        for point in _parse(fileish).points:
            if point.time is not None:
                return point.time
    except (ActivityParseError, XmlSafetyError):
        return None
    return None


def extractIntervals(fileish: Fileish) -> list[dict]:
    """GPX has no lap concept, so there is nothing to extract.

    Returning ``[]`` is the contract ``fit.extractIntervals`` already uses for a
    file with no lap frames: the caller auto-splits.
    """
    return []


def extract_route(fileish: Fileish) -> Route:
    """The ride's geometry, coordinates included — **for in-memory analysis only.**

    Deliberately separate from :func:`summarizeWorkout` so that ingesting an
    activity cannot accidentally acquire location data: the ingestion path calls
    the summariser, which has no way to return coordinates, and a caller wanting
    the route has to say so in the name of the function it calls.

    Intended for the route-analysis work (gradient profiles, climb detection,
    comparing two rides over the same course), which computes from the geometry
    and keeps the conclusions. See :class:`Route` for why the geometry itself
    must not be persisted.
    """
    track = _parse(fileish)
    points = track.points
    distance, _ = _derive(points)

    start_time = next((p.time for p in points if p.time is not None), None)

    route_points: list[RoutePoint] = []
    for point in points:
        if not _has_fix(point):
            continue
        offset = None
        if start_time is not None and point.time is not None:
            offset = int((point.time - start_time).total_seconds())
        route_points.append(
            RoutePoint(
                latitude=point.lat,
                longitude=point.lon,
                elevation_m=point.elevation,
                offset_s=offset,
            )
        )

    return Route(
        points=route_points,
        start_time=start_time,
        name=track.name,
        sport_type=_normalise_sport(track.sport_type),
        distance_m=distance,
        elevation_gain_m=geo.elevation_gain_m(p.elevation for p in points),
    )
