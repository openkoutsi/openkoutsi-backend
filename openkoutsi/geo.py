"""Distance and elevation derived from a coordinate track.

GPX is *made of* coordinates, and openkoutsi stores none: the promise in
``openkoutsi-docs/docs/data-and-ai.md`` is that no location data is kept, and
``scripts/strip_fit_location.py`` exists to hold FIT files to it. The parsers in
:mod:`openkoutsi.gpx` and :mod:`openkoutsi.tcx` therefore read coordinates only
to produce the two scalars a training file would otherwise carry itself —
distance and elevation gain — and drop them before anything is returned to a
caller that could persist it.

This module is that derivation, kept separate so it is obvious there is exactly
one place coordinates are consumed and nothing here holds onto them.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

# IUGG mean Earth radius. Haversine on a sphere is good to ~0.5% against the
# ellipsoid, which is well inside the error a consumer-grade GPS contributes to
# each fix anyway — and unlike a Vincenty solution it cannot fail to converge on
# the near-antipodal garbage a corrupt file can contain.
EARTH_RADIUS_M = 6371008.8

# Climb smaller than this is noise, not ascent. A stationary GPS wanders several
# metres vertically, so summing every positive delta over a three-hour ride
# accumulates hundreds of phantom metres — the classic reason a GPX ride shows
# more climbing than the head unit that recorded it. The threshold is applied as
# hysteresis (see `elevation_gain_m`) rather than per-sample, so a genuine long
# drag is still counted in full.
_ELEVATION_NOISE_M = 3.0

# Samples averaged before the threshold is applied. A threshold alone cannot
# separate a 4 m oscillation at 0.05 Hz (a receiver sitting still) from a 4 m
# roller, because they are the same amplitude — what tells them apart is that
# noise is much faster than terrain. Fifteen seconds of averaging flattens the
# first and leaves the second, and is why the threshold can stay small enough to
# keep short climbs.
_ELEVATION_SMOOTHING_SAMPLES = 15

# 216 km/h. A step whose implied speed is above this is a wrong pair of fixes,
# not a fast one — the same rule the parsers already apply to a derived speed
# stream, applied here to the distance that speed came from.
MAX_STEP_SPEED_MS = 60.0

# The same glitch rule for a track that carries no time: a course file, a route
# export, or a point whose `<time>` was missing or unparseable. A cap in metres
# alone cannot tell travel from a glitch — that depends entirely on how long the
# step took — so it is set where *catastrophe* begins rather than where "far"
# does: a receiver reacquiring after a tunnel lands within a few km, while a
# corrupt fix lands thousands of km away or at (0, 0), which `_valid` already
# drops. Prefer :func:`step_is_travel` with a real ``dt_s`` wherever one exists.
MAX_STEP_M = 10_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points, in metres."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, a)))


def _valid(point: tuple[float, float] | None) -> bool:
    if point is None:
        return False
    lat, lon = point
    if lat is None or lon is None:
        return False
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return False
    # (0, 0) is in the Gulf of Guinea and is what a device with no fix writes.
    if lat == 0.0 and lon == 0.0:
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def step_is_travel(step_m: float, dt_s: float | None) -> bool:
    """Did the athlete ride these metres, or did the receiver jump?

    The one place that decision is made, because getting it wrong is silent in
    both directions: too permissive and a glitch adds phantom kilometres, too
    strict and real ones are subtracted from a total nobody can check.

    Where the track carries time, the test is the implied **speed**: a step is
    travel unless it would have taken more than :data:`MAX_STEP_SPEED_MS`. That
    is the only test that holds for every recording rate, and recording rate is
    not something a file has to tell us. A 1 Hz head unit writes a point every
    few metres; smart recording writes one every few hundred; a route planner
    exports a vertex per direction change, which on a long straight puts
    kilometres between consecutive points. All three are the same ride.

    Where it does not — a course, a route export, a point with no parseable
    ``<time>`` — there is nothing to compute a speed from and
    :data:`MAX_STEP_M` is the fallback.
    """
    if dt_s is not None and dt_s > 0:
        return step_m <= dt_s * MAX_STEP_SPEED_MS
    return step_m <= MAX_STEP_M


def cumulative_distance_m(
    points: Sequence[tuple[float, float] | None],
    elapsed_s: Sequence[float | None] | None = None,
) -> list[float]:
    """Running distance along a track, one entry per point.

    Points without a fix (``None``, out of range, or the (0, 0) a device writes
    before it locks on) carry the distance forward unchanged rather than
    dropping out, so index ``i`` of the result still describes point ``i``.
    Steps that are not travel (see :func:`step_is_travel`) are skipped.

    ``elapsed_s``, when given, is seconds from an arbitrary origin for each
    point — one entry per point, ``None`` where the point had no time. Passing
    it is what lets the glitch rule be about speed rather than about distance;
    without it every step falls back to :data:`MAX_STEP_M`, which on a track
    recorded at anything other than ~1 Hz will throw away real kilometres.
    """
    if elapsed_s is not None and len(elapsed_s) != len(points):
        raise ValueError("elapsed_s must be the same length as points")

    out: list[float] = []
    total = 0.0
    previous: tuple[float, float] | None = None
    previous_time: float | None = None
    for index, point in enumerate(points):
        now = elapsed_s[index] if elapsed_s is not None else None
        if _valid(point):
            if previous is not None:
                step = haversine_m(previous[0], previous[1], point[0], point[1])
                dt = None if (now is None or previous_time is None) else now - previous_time
                if step_is_travel(step, dt):
                    total += step
            previous = point  # type: ignore[assignment]
            previous_time = now
        out.append(total)
    return out


def track_distance_m(
    points: Sequence[tuple[float, float] | None],
    elapsed_s: Sequence[float | None] | None = None,
) -> float:
    """Total distance along a coordinate track, in metres."""
    cumulative = cumulative_distance_m(points, elapsed_s)
    return cumulative[-1] if cumulative else 0.0


def _smoothed(values: Sequence[float], window: int) -> list[float]:
    """Centred moving average, with the window shrinking at both ends."""
    if window <= 1 or len(values) < 3:
        return list(values)
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)

    half = window // 2
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append((prefix[hi] - prefix[lo]) / (hi - lo))
    return out


def smoothed_by_distance(
    values: Sequence[float],
    distances_m: Sequence[float],
    window_m: float,
) -> list[float]:
    """Centred moving average over a *distance* window rather than a sample count.

    :func:`_smoothed` averages a fixed number of samples, which is right for a
    1 Hz activity stream where samples are evenly spaced in time. A course has
    no clock: its points are spaced by metres, and unevenly, so "fifteen
    samples" means 100 m on one stretch and 2 km on another. Here the window is
    ``window_m`` metres of track centred on each point — every value whose
    distance lies within ``window_m / 2`` of the point's own contributes.

    ``distances_m`` must be non-decreasing (a running distance along the track,
    as :func:`cumulative_distance_m` produces) and the same length as
    ``values``. Like :func:`_smoothed`, the window shrinks at the ends of the
    series, and a window that never spans more than one point returns the
    input unchanged.
    """
    n = len(values)
    if n != len(distances_m):
        raise ValueError("values and distances_m must be the same length")
    if n < 3 or window_m <= 0.0:
        return list(values)

    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)

    half = window_m / 2.0
    out: list[float] = []
    lo = 0
    hi = 0
    for i in range(n):
        centre = distances_m[i]
        while distances_m[lo] < centre - half:
            lo += 1
        if hi < i:
            hi = i
        while hi + 1 < n and distances_m[hi + 1] <= centre + half:
            hi += 1
        out.append((prefix[hi + 1] - prefix[lo]) / (hi + 1 - lo))
    return out


def elevation_gain_m(
    altitudes: Iterable[float | None],
    *,
    threshold_m: float = _ELEVATION_NOISE_M,
    smoothing: int = _ELEVATION_SMOOTHING_SAMPLES,
) -> float:
    """Total ascent over an elevation series, in metres.

    Two stages, because either alone gets a real ride wrong:

    * **Smooth**, to separate terrain from receiver noise by how fast it moves.
    * **Accumulate with hysteresis**, which needs ``threshold_m`` of rise to
      *start* counting a climb but then counts every metre of it until an equal
      descent ends it. Confirming the climb before crediting it is what rejects
      oscillation; crediting it continuously afterwards is what stops a long
      drag from losing a slice of itself at every step.

    Gaps are skipped rather than interpolated: a channel that stopped recording
    for a minute contributes the climb either side of the hole and nothing for
    the hole itself. The smoothing window shrinks for short series so that a
    handful of samples is not averaged into a flat line.
    """
    values = [
        float(v) for v in altitudes if v is not None and math.isfinite(v)
    ]
    if len(values) < 2:
        return 0.0

    window = min(smoothing, max(1, len(values) // 4))
    series = _smoothed(values, window)

    # `reference` is the lowest point seen since the last confirmed climb while
    # not climbing, and the highest point of the current climb while climbing.
    reference = series[0]
    climbing = False
    gain = 0.0

    for value in series[1:]:
        if climbing:
            if value >= reference:
                gain += value - reference
                reference = value
            elif reference - value >= threshold_m:
                climbing = False
                reference = value
            # A dip smaller than the threshold is noise inside the climb: hold
            # the reference so the re-rise is not banked a second time.
        elif value < reference:
            reference = value
        elif value - reference >= threshold_m:
            gain += value - reference
            reference = value
            climbing = True

    return gain
