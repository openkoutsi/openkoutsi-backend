"""Course recon — gradient segmentation and pacing physics for a GPX course (issue #55).

A *course* is a route the athlete is going to ride, uploaded on purpose for
pacing analysis — as opposed to an activity, which records a ride that already
happened. The pipeline here turns a parsed :class:`openkoutsi.gpx.Route` into
a segment table with a power target and predicted split per segment, solved
from the athlete's own FTP and weight plus a handful of bike parameters:

1. :func:`thin_track` — reduce the raw track to ~8 m point spacing.
2. :func:`course_profile` — smooth elevation over a *distance* window and
   derive gradient. This is the boundary where coordinates are dropped:
   everything downstream is distance/elevation/gradient only.
3. :func:`segment_by_gradient` — split where gradient meaningfully changes,
   dissolving runs too short to be worth a segment.
4. :func:`solve_speed_ms` / :func:`predict_splits` — the steady-state power
   balance ``P·η = v·m·g·(Crr·cosθ + sinθ) + ½·ρ·CdA·v³`` solved for speed.
5. :func:`solve_target_time` — the inverse: distribute effort to hit a
   requested finish time, refusing (with a reason code, not a number) targets
   that would take more power than a human sustains.
6. :func:`solve_target_power` — the other inverse: hold a requested *average*
   power and report the time it produces. Same effort distribution, the other
   half of the question an athlete actually asks.

Everything here is pure — no DB, no I/O, plain floats in and out. The physics
carries a **still-air, dry-pavement assumption**: wind is Stage 3 (#57) and
surface classification is Stage 2 (#56), and until then a plan built from
these numbers should say so rather than imply a windless day is a prediction.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from . import geo
from .gpx import Route

# ── physical constants ────────────────────────────────────────────────────────

# Standard gravity. The variation over rideable latitudes and altitudes is
# ~0.3%, far below the uncertainty in any of the aerodynamic inputs.
GRAVITY_MS2 = 9.80665

# Sea-level air density at ~20 °C. Stage 1 takes no altitude or temperature
# input; at alpine altitude the true value is ~10% lower, which flatters
# nobody's climb but errs on the conservative side for the splits.
AIR_DENSITY_KGM3 = 1.20

# Chain and derailleur losses. Measured drivetrain loss on a clean, lubricated
# road drivetrain is 2–5% (Spicer et al. 2001); 2.5% assumes the athlete shows
# up with a bike that has seen a rag recently.
DRIVETRAIN_EFFICIENCY = 0.975

# Bike-plus-kit mass added to the athlete's own weight: ~8 kg road bike plus
# ~2 kg of bottles, spares, helmet and clothing. A per-bike weight field can
# replace this allowance later without touching the physics.
BIKE_AND_KIT_MASS_KG = 10.0

# ── ingest and profile constants ──────────────────────────────────────────────

# Target point spacing after thinning. Fine enough that a 100 m feature still
# spans ~12 points, coarse enough that a 200 km course is ~25k points and the
# per-interval gradient noise from ±1 m elevation error is bounded before
# smoothing ever runs.
THIN_SPACING_M = 8.0

# Ceiling on the thinned point count, whatever the spacing implies. At 8 m
# this is ~320 km, which covers any course anyone rides in a day; beyond it the
# spacing widens so an absurdly long upload degrades in resolution rather than
# in size. The stored track is one JSON row that re-analysis re-materialises in
# full, so an unbounded point count is an unbounded row — a 3000 km GPX
# produced a 14 MB one. Segmentation is unaffected in practice: the widened
# spacing stays far below the 200 m minimum segment.
MAX_THINNED_POINTS = 40_000

# Elevation is smoothed over this many metres of track before any gradient is
# computed. DEM and barometric elevation are noisy at the 1–3 m level; over an
# 8 m step that is tens of percent of instantaneous gradient, which a 60 m
# window averages to well under 1% while a 200 m climb keeps most of its true
# gradient. Distance-based rather than time-based because a course has no
# clock — see issue #39, which calls naive differencing "the single biggest
# correctness risk", a warning that transfers verbatim.
ELEVATION_SMOOTHING_WINDOW_M = 60.0

# Gradient is a centred difference of the *smoothed* elevation over at least
# this much track, not over adjacent points: an 8–16 m baseline divides the
# smoothing residual by a small number and swings percentage points, while a
# 40 m baseline is still far shorter than anything worth a segment.
GRADIENT_BASELINE_M = 40.0

# A course shorter than this has fewer gradient features than the smoothing
# window itself and produces no meaningful segmentation.
MIN_COURSE_LENGTH_M = 500.0

# ── segmentation constants ────────────────────────────────────────────────────

# A new segment starts when the local gradient departs from the running
# segment's length-weighted mean by at least this much (2 percentage points) —
# about the smallest gradient difference that changes pacing advice.
SEGMENT_SPLIT_GRADE = 0.02

# Runs shorter than this are dissolved into the neighbouring segment with the
# closer mean gradient. ~20 s of riding is below actionable pacing
# granularity, and without the dissolve a rolling road shatters into
# hundreds of rows.
MIN_SEGMENT_LENGTH_M = 200.0

# |mean gradient| below this is "flat"; at or above it, "climb" or "descent".
GRADE_FLAT_MAX = 0.02

# Adjacent climb segments whose combined gain reaches this are reported as one
# named "key climb" feature — the level of thing a written plan calls out.
CLIMB_FEATURE_MIN_GAIN_M = 30.0

# ── rider / bike parameter tables ─────────────────────────────────────────────

# CdA (m²) by riding position, from wind-tunnel and field literature (Martin
# et al. 1998; Wilson, *Bicycling Science*). Coarse buckets on purpose: the
# athlete knows "drops" or "hoods"; nobody knows their CdA to two decimals.
CDA_BY_POSITION_M2 = {
    "tops": 0.40,
    "hoods": 0.36,
    "drops": 0.32,
    "aero": 0.27,
}
DEFAULT_POSITION = "hoods"

# ── effort model constants ────────────────────────────────────────────────────

# Power weighting by gradient for split prediction: w(g) = 1 + 4·g, clamped.
# +4% power per 1% of gradient encodes the standard result that time is won on
# climbs (speed is lowest, so a watt buys the most seconds) and unspendable on
# descents. The clamp keeps the model from prescribing sprint power on walls
# or negative power on drops.
GRADE_POWER_GAIN = 4.0
EFFORT_WEIGHT_MIN = 0.60
EFFORT_WEIGHT_MAX = 1.15

# Bisection bracket for the intensity k (fraction of FTP the effort
# model scales from). 0.30 is soft-pedalling; 1.20 is above anyone's hour
# power and exists only so the solver can prove a target impossible.
INTENSITY_MIN = 0.30
INTENSITY_MAX = 1.20

# Real descents are capped by braking, corners and self-preservation long
# before still-air terminal velocity. 60 km/h; segments the cap binds on are
# reported as coasting, because on them time is set by nerve, not watts.
DESCENT_SPEED_CAP_MS = 60.0 / 3.6

# Floor for split arithmetic so a pathological segment (20% wall at
# soft-pedal power) yields a long split rather than a division blow-up.
_MIN_SPEED_MS = 0.1

# The intensity a plan defaults to when no target time is given: firmly
# aerobic, and for long days pulled down to a steady fraction of the
# sustainable ceiling rather than riding at the ceiling itself.
DEFAULT_INTENSITY = 0.75
DEFAULT_INTENSITY_OF_CEILING = 0.85

# Sustainable-intensity model: FTP is roughly one-hour power, so an intensity
# of 1.0 is credible up to an hour and decays ~0.05 per doubling of duration —
# matching the familiar ~0.85 of FTP for a 5 h event and ~0.75 for 10 h.
SUSTAINABLE_INTENSITY_FLOOR = 0.70


def max_sustainable_intensity(duration_s: float) -> float:
    """The largest intensity plausibly sustainable for ``duration_s``."""
    if duration_s <= 3600.0:
        return 1.0
    return max(SUSTAINABLE_INTENSITY_FLOOR, 1.0 - 0.05 * math.log2(duration_s / 3600.0))


def crr_for_tyre_width(width_mm: int | None) -> float:
    """Rolling-resistance coefficient on pavement, by tyre width.

    Measured Crr for road tyres spans roughly 0.003–0.007
    (bicyclerollingresistance.com drum tests). Width is a workable proxy: the
    26–32 mm range rolls best at sensible pressures, narrower gives a little
    back on real road surfaces, and wider means heavier casings. Stage 1
    assumes pavement throughout — the plan prose says so — and Stage 2 (#56)
    replaces this with a per-surface value.
    """
    if width_mm is None:
        return 0.0045
    if width_mm <= 25:
        return 0.0045
    if width_mm <= 32:
        return 0.0042
    if width_mm <= 45:
        return 0.0050
    return 0.0065


# ── dataclasses ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrackPoint:
    """One thinned course point — coordinates included."""

    latitude: float
    longitude: float
    distance_m: float
    elevation_m: float | None


@dataclass(frozen=True)
class CourseTrack:
    """The thinned course geometry — **the quarantine type for coordinates.**

    Exists for exactly two consumers: persistence (the course's stored track,
    per the decision in issue #54) and :func:`course_profile`, which converts
    it into the coordinate-free profile everything else runs on.
    :func:`analyze_course` deliberately does not accept one of these, so the
    analysis, storage-row and prompt pipeline downstream of the single
    conversion call is typed coordinate-free. Never format one into a string,
    a prompt, or an API response — ``repr`` shows only the point count on
    purpose.
    """

    points: list[TrackPoint] = field(repr=False)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"CourseTrack({len(self.points)} points)"

    @property
    def total_distance_m(self) -> float:
        return self.points[-1].distance_m if self.points else 0.0


@dataclass(frozen=True)
class ProfilePoint:
    """One point of the coordinate-free course profile."""

    distance_m: float
    elevation_m: float  # smoothed
    gradient: float  # fraction; centred difference of the smoothed elevation


@dataclass(frozen=True)
class CourseProfile:
    """What the analysis runs on: distance, smoothed elevation, gradient."""

    points: list[ProfilePoint]
    total_distance_m: float


@dataclass(frozen=True)
class RiderParams:
    ftp_w: float
    weight_kg: float


@dataclass(frozen=True)
class BikeParams:
    tyre_width_mm: int | None
    riding_position: str  # a key of CDA_BY_POSITION_M2

    @property
    def cda_m2(self) -> float:
        return CDA_BY_POSITION_M2.get(self.riding_position, CDA_BY_POSITION_M2[DEFAULT_POSITION])

    @property
    def crr(self) -> float:
        return crr_for_tyre_width(self.tyre_width_mm)


@dataclass(frozen=True)
class Segment:
    index: int
    start_distance_m: float
    end_distance_m: float
    length_m: float
    avg_gradient: float
    elevation_change_m: float
    segment_type: str  # "climb" | "flat" | "descent"


@dataclass(frozen=True)
class SegmentPlan:
    """A segment plus the physics outputs for one intensity."""

    segment: Segment
    power_w: float
    speed_ms: float
    duration_s: float
    start_offset_s: float
    speed_capped: bool


@dataclass(frozen=True)
class ClimbFeature:
    """Adjacent climb segments grouped into one feature worth naming in prose."""

    start_distance_m: float
    length_m: float
    avg_gradient: float
    elevation_gain_m: float
    duration_s: float | None
    avg_power_w: float | None


@dataclass(frozen=True)
class PacingSolution:
    feasible: bool
    refusal_reason: str | None  # None | "target_faster_than_physics" | "exceeds_sustainable_power"
    intensity: float | None
    required_intensity: float | None
    predicted_time_s: float | None
    splits: list[SegmentPlan]


@dataclass(frozen=True)
class CourseAnalysis:
    """The complete, coordinate-free result of analysing a course."""

    total_distance_m: float
    elevation_gain_m: float
    elevation_loss_m: float
    min_elevation_m: float
    max_elevation_m: float
    profile: list[ProfilePoint]  # downsampled for charting
    segments: list[Segment]
    pacing: PacingSolution


# ── ingest ────────────────────────────────────────────────────────────────────


def thin_track(route: Route, spacing_m: float = THIN_SPACING_M) -> CourseTrack:
    """Reduce a parsed route to roughly ``spacing_m`` point spacing.

    Distance accumulates by haversine with the same :data:`geo.MAX_STEP_M`
    glitch rule the activity path uses. The first and last points are always
    kept, and the spacing widens if ``spacing_m`` would produce more than
    :data:`MAX_THINNED_POINTS` of them. Points missing elevation are filled by linear interpolation over
    distance between their nearest elevated neighbours (ends take the nearest
    known value); a route with no elevation at all keeps ``None`` throughout
    and is rejected later by :func:`course_profile` with a reason code.
    """
    # Measure first, so the spacing can be widened before anything is kept.
    # Deliberately re-derived here rather than read off ``Route.distance_m``:
    # only the parser populates that, and a cap that quietly does nothing for
    # a caller who built the route another way is not a cap.
    cumulative: list[float] = []
    total = 0.0
    prev: tuple[float, float] | None = None
    for point in route.points:
        if prev is not None:
            step = geo.haversine_m(prev[0], prev[1], point.latitude, point.longitude)
            if step <= geo.MAX_STEP_M:
                total += step
        prev = (point.latitude, point.longitude)
        cumulative.append(total)

    if total > 0:
        spacing_m = max(spacing_m, total / MAX_THINNED_POINTS)

    kept: list[TrackPoint] = []
    last_kept_at = -math.inf
    final_index = len(route.points) - 1

    for index, point in enumerate(route.points):
        travelled = cumulative[index]
        # Positional, not `is`: a route may repeat the same point object (a
        # closed loop written with its start coordinate again), and identity
        # would then call an early point the last one.
        is_last = index == final_index
        if travelled - last_kept_at >= spacing_m or not kept or is_last:
            if kept and travelled - kept[-1].distance_m < 0.5:
                # Too close to the previous kept point to bound a meaningful
                # gradient interval; keep whichever came later.
                kept.pop()
            kept.append(
                TrackPoint(
                    latitude=point.latitude,
                    longitude=point.longitude,
                    distance_m=travelled,
                    elevation_m=point.elevation_m,
                )
            )
            last_kept_at = travelled

    return CourseTrack(points=_interpolate_missing_elevation(kept))


def _interpolate_missing_elevation(points: list[TrackPoint]) -> list[TrackPoint]:
    known = [(i, p.elevation_m) for i, p in enumerate(points) if p.elevation_m is not None]
    if not known or len(known) == len(points):
        return points

    out = list(points)
    # Ends take the nearest known value.
    first_i, first_e = known[0]
    for i in range(first_i):
        out[i] = _with_elevation(out[i], first_e)
    last_i, last_e = known[-1]
    for i in range(last_i + 1, len(out)):
        out[i] = _with_elevation(out[i], last_e)
    # Interior gaps interpolate linearly over distance.
    for (i0, e0), (i1, e1) in zip(known, known[1:]):
        d0, d1 = points[i0].distance_m, points[i1].distance_m
        span = d1 - d0
        for i in range(i0 + 1, i1):
            frac = (points[i].distance_m - d0) / span if span > 0 else 0.5
            out[i] = _with_elevation(out[i], e0 + frac * (e1 - e0))
    return out


def _with_elevation(point: TrackPoint, elevation_m: float) -> TrackPoint:
    return TrackPoint(
        latitude=point.latitude,
        longitude=point.longitude,
        distance_m=point.distance_m,
        elevation_m=elevation_m,
    )


def course_profile(track: CourseTrack) -> tuple[CourseProfile | None, str | None]:
    """The coordinate-free profile of a thinned track, or a reason it has none.

    Returns ``(profile, None)`` on success, or ``(None, reason)`` with reason
    ``"no_elevation_data"`` or ``"course_too_short"`` — a stable code rather
    than an exception, following the convention that "not analysable" is a
    normal outcome with a stated cause (see
    ``training_math.decoupling_unavailable_reason``).

    This is the only function that consumes a :class:`CourseTrack`; everything
    downstream sees distance, elevation and gradient only.
    """
    points = track.points
    if len(points) < 2 or track.total_distance_m < MIN_COURSE_LENGTH_M:
        return None, "course_too_short"
    if any(p.elevation_m is None for p in points):
        return None, "no_elevation_data"

    distances = [p.distance_m for p in points]
    elevations = [float(p.elevation_m) for p in points]  # type: ignore[arg-type]
    smoothed = geo.smoothed_by_distance(elevations, distances, ELEVATION_SMOOTHING_WINDOW_M)

    n = len(points)
    half = GRADIENT_BASELINE_M / 2.0
    profile: list[ProfilePoint] = []
    lo = 0
    hi = 0
    for i in range(n):
        centre = distances[i]
        while lo + 1 < n and distances[lo + 1] <= centre - half:
            lo += 1
        if hi < i:
            hi = i
        while hi + 1 < n and distances[hi + 1] <= centre + half + 1e-9:
            hi += 1
        a = min(lo, i - 1) if i > 0 else lo
        b = max(hi, i + 1) if i < n - 1 else hi
        a = max(a, 0)
        b = min(b, n - 1)
        span = distances[b] - distances[a]
        gradient = (smoothed[b] - smoothed[a]) / span if span > 0 else 0.0
        profile.append(
            ProfilePoint(distance_m=distances[i], elevation_m=smoothed[i], gradient=gradient)
        )
    return CourseProfile(points=profile, total_distance_m=distances[-1]), None


# ── segmentation ──────────────────────────────────────────────────────────────


def segment_by_gradient(profile: CourseProfile) -> list[Segment]:
    """Split the course where gradient meaningfully changes.

    Greedy first pass: an interval joins the running segment while its
    gradient stays within :data:`SEGMENT_SPLIT_GRADE` of the segment's
    length-weighted mean. Dissolve pass: segments shorter than
    :data:`MIN_SEGMENT_LENGTH_M` merge into the neighbour with the closer
    mean gradient, repeatedly, so noise cannot shatter the route into
    hundreds of rows. Segments tile the course exactly.
    """
    pts = profile.points
    if len(pts) < 2:
        return []

    # (start_d, end_d, elevation_change) per raw segment.
    raw: list[list[float]] = []
    for a, b in zip(pts, pts[1:]):
        length = b.distance_m - a.distance_m
        if length <= 0:
            continue
        rise = b.elevation_m - a.elevation_m
        grad = rise / length
        if raw:
            seg = raw[-1]
            seg_len = seg[1] - seg[0]
            seg_grad = seg[2] / seg_len if seg_len > 0 else 0.0
            if abs(grad - seg_grad) <= SEGMENT_SPLIT_GRADE:
                seg[1] = b.distance_m
                seg[2] += rise
                continue
        raw.append([a.distance_m, b.distance_m, rise])

    # Dissolve short segments into the gradient-closer neighbour.
    def _grad(seg: list[float]) -> float:
        length = seg[1] - seg[0]
        return seg[2] / length if length > 0 else 0.0

    # Dissolve over a linked list, resuming at the merged segment rather than
    # restarting the scan. Restarting from zero after every merge — and
    # deleting from the middle of a list — makes this quadratic in the raw
    # segment count, which is worst exactly on the rolling terrain the dissolve
    # exists for: a 400 km course spent ~100 s here. Resuming is equivalent
    # because everything before the merge point was already checked in this
    # same traversal and the merge does not change it.
    n = len(raw)
    prev = list(range(-1, n - 1))
    nxt = list(range(1, n + 1))
    if nxt:
        nxt[-1] = -1
    dead = [False] * n

    i = 0 if n else -1
    while i != -1:
        seg = raw[i]
        if seg[1] - seg[0] >= MIN_SEGMENT_LENGTH_M:
            i = nxt[i]
            continue
        p, q = prev[i], nxt[i]
        if p == -1 and q == -1:
            break  # the only segment left: nothing to dissolve into
        if p == -1:
            target = q
        elif q == -1:
            target = p
        else:
            # Ties go to the lower index, as `min` over [i-1, i+1] did.
            target = p if abs(_grad(raw[p]) - _grad(seg)) <= abs(_grad(raw[q]) - _grad(seg)) else q
        lo, hi = (p, i) if target == p else (i, q)
        raw[lo] = [raw[lo][0], raw[hi][1], raw[lo][2] + raw[hi][2]]
        dead[hi] = True
        nxt[lo] = nxt[hi]
        if nxt[hi] != -1:
            prev[nxt[hi]] = lo
        i = lo

    raw = [raw[k] for k in range(n) if not dead[k]]

    segments: list[Segment] = []
    for index, seg in enumerate(raw):
        length = seg[1] - seg[0]
        grad = _grad(seg)
        if grad >= GRADE_FLAT_MAX:
            kind = "climb"
        elif grad <= -GRADE_FLAT_MAX:
            kind = "descent"
        else:
            kind = "flat"
        segments.append(
            Segment(
                index=index,
                start_distance_m=seg[0],
                end_distance_m=seg[1],
                length_m=length,
                avg_gradient=grad,
                elevation_change_m=seg[2],
                segment_type=kind,
            )
        )
    return segments


# ── physics ───────────────────────────────────────────────────────────────────


def solve_speed_ms(
    power_w: float,
    gradient: float,
    total_mass_kg: float,
    crr: float,
    cda_m2: float,
    *,
    air_density: float = AIR_DENSITY_KGM3,
    efficiency: float = DRIVETRAIN_EFFICIENCY,
) -> float:
    """Steady-state speed for a given power, uncapped.

    Solves ``P·η = v·m·g·(Crr·cosθ + sinθ) + ½·ρ·CdA·v³`` by bisection —
    matching the house preference for a bracketing method over a continuous
    optimiser (see the rationale in ``training_math``, which measured and
    rejected scipy). The demand side is eventually strictly increasing in v,
    so the bracket [0, 40 m/s] contains exactly one crossing for any P ≥ 0;
    on a descent, P = 0 yields the still-air terminal velocity. Callers apply
    :data:`DESCENT_SPEED_CAP_MS` — this function reports the physics.
    """
    theta = math.atan(gradient)
    linear = total_mass_kg * GRAVITY_MS2 * (crr * math.cos(theta) + math.sin(theta))
    cubic = 0.5 * air_density * cda_m2
    supply = max(0.0, power_w) * efficiency

    def demand(v: float) -> float:
        return linear * v + cubic * v**3

    lo, hi = 0.0, 40.0
    if demand(hi) <= supply:
        return hi  # Beyond any plausible bracket; the descent cap will bind.
    if linear >= 0.0 and supply <= 0.0:
        return 0.0  # No power on flat or uphill: stationary.
    # On a descent (linear < 0) demand dips below zero before rising, so for
    # any supply ≥ 0 the interior of the bracket sits below the supply until
    # the single rising crossing — bisection lands on the moving root (the
    # terminal velocity when supply is zero), never on v = 0.
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if demand(mid) < supply:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _effort_weight(gradient: float) -> float:
    return min(EFFORT_WEIGHT_MAX, max(EFFORT_WEIGHT_MIN, 1.0 + GRADE_POWER_GAIN * gradient))


def predict_splits(
    segments: Sequence[Segment],
    rider: RiderParams,
    bike: BikeParams,
    intensity: float,
) -> list[SegmentPlan]:
    """Per-segment power targets and predicted splits at one intensity.

    Segment power is ``k · FTP · w(gradient)`` with the clamped effort weight
    — spend on the climbs, recover on the descents. Where even zero power
    exceeds the descent speed cap, the segment is reported as coasting at the
    cap: on such roads the model has nothing honest to say about watts.
    """
    total_mass = rider.weight_kg + BIKE_AND_KIT_MASS_KG
    crr = bike.crr
    cda = bike.cda_m2

    plans: list[SegmentPlan] = []
    offset = 0.0
    for segment in segments:
        power = intensity * rider.ftp_w * _effort_weight(segment.avg_gradient)
        coasting_speed = solve_speed_ms(0.0, segment.avg_gradient, total_mass, crr, cda)
        if coasting_speed >= DESCENT_SPEED_CAP_MS:
            power, speed, capped = 0.0, DESCENT_SPEED_CAP_MS, True
        else:
            speed = solve_speed_ms(power, segment.avg_gradient, total_mass, crr, cda)
            capped = speed >= DESCENT_SPEED_CAP_MS
            speed = min(speed, DESCENT_SPEED_CAP_MS)
        duration = segment.length_m / max(speed, _MIN_SPEED_MS)
        plans.append(
            SegmentPlan(
                segment=segment,
                power_w=power,
                speed_ms=speed,
                duration_s=duration,
                start_offset_s=offset,
                speed_capped=capped,
            )
        )
        offset += duration
    return plans


def _total_time_s(plans: Sequence[SegmentPlan]) -> float:
    return sum(plan.duration_s for plan in plans)


def _required_intensity(plans: Sequence[SegmentPlan], ftp_w: float) -> float:
    """Time-weighted average power over the ride, as a fraction of FTP."""
    total = _total_time_s(plans)
    if total <= 0 or ftp_w <= 0:
        return 0.0
    return sum(plan.power_w * plan.duration_s for plan in plans) / total / ftp_w


def _unsolvable() -> PacingSolution:
    """The answer when there is nothing to solve — no segments, or no FTP.

    Shared by all three solvers so the degenerate case has one shape. The
    reason code is the "no ride exists here" one; it is not reachable through
    the API, which refuses a too-short course and a profile without FTP and
    weight long before a solver is called.
    """
    return PacingSolution(
        feasible=False,
        refusal_reason="target_faster_than_physics",
        intensity=None,
        required_intensity=None,
        predicted_time_s=None,
        splits=[],
    )


def solve_target_time(
    segments: Sequence[Segment],
    rider: RiderParams,
    bike: BikeParams,
    target_time_s: float,
) -> PacingSolution:
    """Distribute effort to hit a requested finish time — or refuse.

    Total time is monotone non-increasing in the intensity (capped
    descents contribute a k-independent floor, which preserves monotonicity),
    so bisection over k is safe. Refusals are reason codes, not numbers:

    * ``"target_faster_than_physics"`` — quicker than even k = 1.20 delivers.
    * ``"exceeds_sustainable_power"`` — reachable on paper, but only at an
      average intensity no one holds for that long (``required_intensity`` reports
      what it would take, against :func:`max_sustainable_intensity`).

    A target slower than the easiest ride is not refused: the plan clamps to
    the minimum intensity and simply finishes early.
    """
    if not segments or rider.ftp_w <= 0:
        return _unsolvable()

    fastest = predict_splits(segments, rider, bike, INTENSITY_MAX)
    if target_time_s < _total_time_s(fastest):
        return PacingSolution(
            feasible=False,
            refusal_reason="target_faster_than_physics",
            intensity=None,
            required_intensity=_required_intensity(fastest, rider.ftp_w),
            predicted_time_s=_total_time_s(fastest),
            splits=[],
        )

    easiest = predict_splits(segments, rider, bike, INTENSITY_MIN)
    if target_time_s >= _total_time_s(easiest):
        k = INTENSITY_MIN
        plans = easiest
    else:
        lo, hi = INTENSITY_MIN, INTENSITY_MAX  # time(lo) > target > time(hi)
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if _total_time_s(predict_splits(segments, rider, bike, mid)) > target_time_s:
                lo = mid
            else:
                hi = mid
        k = (lo + hi) / 2.0
        plans = predict_splits(segments, rider, bike, k)

    required = _required_intensity(plans, rider.ftp_w)
    if required > max_sustainable_intensity(target_time_s):
        return PacingSolution(
            feasible=False,
            refusal_reason="exceeds_sustainable_power",
            intensity=k,
            required_intensity=required,
            predicted_time_s=_total_time_s(plans),
            splits=[],
        )
    return PacingSolution(
        feasible=True,
        refusal_reason=None,
        intensity=k,
        required_intensity=required,
        predicted_time_s=_total_time_s(plans),
        splits=plans,
    )


def solve_target_power(
    segments: Sequence[Segment],
    rider: RiderParams,
    bike: BikeParams,
    target_power_w: float,
) -> PacingSolution:
    """Distribute effort around a requested **average** power.

    The mirror image of :func:`solve_target_time`: the athlete fixes the watts
    and the model reports the finish time, rather than fixing the time and
    reporting the watts. What is held to the request is the *time-weighted
    average* over the whole ride — the number on the head unit at the finish —
    not the power of any one segment. The gradient weighting still spends on
    the climbs and backs off on the descents, exactly as it does for a time
    target; asking for 210 W does not mean 210 W everywhere.

    Average power rises with the intensity k — every segment's power is linear
    in k while its duration only shrinks — so the same bisection applies.
    Outside the bracket the request is clamped rather than refused, and
    ``required_intensity`` always reports what the returned plan actually asks
    for, so a clamp shows up as a number that differs from the request instead
    of being silent.

    A power target can never be "faster than physics": it names an effort, and
    an effort is always rideable — the question is only for how long. So the
    one refusal it can earn is ``"exceeds_sustainable_power"``, and it is
    reported **with the splits kept**. That is the deliberate difference from
    :func:`solve_target_time`, which returns none: an impossible time describes
    no ride at all, while an unsustainable power describes a ride the model can
    lay out in full — and the splits are the very thing that shows how long the
    athlete would be holding it.
    """
    if not segments or rider.ftp_w <= 0 or target_power_w <= 0:
        return _unsolvable()

    def _avg_power(plans: Sequence[SegmentPlan]) -> float:
        return _required_intensity(plans, rider.ftp_w) * rider.ftp_w

    hardest = predict_splits(segments, rider, bike, INTENSITY_MAX)
    easiest = predict_splits(segments, rider, bike, INTENSITY_MIN)
    if target_power_w >= _avg_power(hardest):
        k, plans = INTENSITY_MAX, hardest
    elif target_power_w <= _avg_power(easiest):
        k, plans = INTENSITY_MIN, easiest
    else:
        lo, hi = INTENSITY_MIN, INTENSITY_MAX  # avg(lo) < target < avg(hi)
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if _avg_power(predict_splits(segments, rider, bike, mid)) < target_power_w:
                lo = mid
            else:
                hi = mid
        k = (lo + hi) / 2.0
        plans = predict_splits(segments, rider, bike, k)

    predicted = _total_time_s(plans)
    required = _required_intensity(plans, rider.ftp_w)
    sustainable = required <= max_sustainable_intensity(predicted)
    return PacingSolution(
        feasible=sustainable,
        refusal_reason=None if sustainable else "exceeds_sustainable_power",
        intensity=k,
        required_intensity=required,
        predicted_time_s=predicted,
        splits=plans,
    )


def default_pacing(
    segments: Sequence[Segment], rider: RiderParams, bike: BikeParams
) -> PacingSolution:
    """The prediction when no target time is given: a sustainable steady day.

    Seeds at :data:`DEFAULT_INTENSITY`, then lowers once if the resulting
    duration says that intensity would not last the distance.
    """
    if not segments or rider.ftp_w <= 0:
        return _unsolvable()
    plans = predict_splits(segments, rider, bike, DEFAULT_INTENSITY)
    k = min(
        DEFAULT_INTENSITY,
        DEFAULT_INTENSITY_OF_CEILING * max_sustainable_intensity(_total_time_s(plans)),
    )
    if k < DEFAULT_INTENSITY:
        plans = predict_splits(segments, rider, bike, k)
    return PacingSolution(
        feasible=True,
        refusal_reason=None,
        intensity=k,
        required_intensity=_required_intensity(plans, rider.ftp_w),
        predicted_time_s=_total_time_s(plans),
        splits=plans,
    )


def key_climbs(plans: Sequence[SegmentPlan]) -> list[ClimbFeature]:
    """Adjacent climb segments grouped into features worth naming in prose.

    Groups runs of consecutive ``"climb"`` segments and keeps those whose
    combined gain reaches :data:`CLIMB_FEATURE_MIN_GAIN_M`.
    """
    features: list[ClimbFeature] = []
    run: list[SegmentPlan] = []

    def _flush() -> None:
        if not run:
            return
        gain = sum(p.segment.elevation_change_m for p in run)
        if gain < CLIMB_FEATURE_MIN_GAIN_M:
            return
        length = sum(p.segment.length_m for p in run)
        duration = sum(p.duration_s for p in run)
        features.append(
            ClimbFeature(
                start_distance_m=run[0].segment.start_distance_m,
                length_m=length,
                avg_gradient=gain / length if length > 0 else 0.0,
                elevation_gain_m=gain,
                duration_s=duration if duration > 0 else None,
                avg_power_w=(
                    sum(p.power_w * p.duration_s for p in run) / duration
                    if duration > 0
                    else None
                ),
            )
        )

    for plan in plans:
        if plan.segment.segment_type == "climb":
            run.append(plan)
        else:
            _flush()
            run = []
    _flush()
    return features


# ── orchestration ─────────────────────────────────────────────────────────────

# The chart payload is capped here: 400 distance-even points draw an elevation
# profile indistinguishable from the full series at any plausible screen width.
CHART_PROFILE_MAX_POINTS = 400


def _resample_profile(points: Sequence[ProfilePoint]) -> list[ProfilePoint]:
    """The chart payload: at most :data:`CHART_PROFILE_MAX_POINTS` samples on an
    **evenly spaced** distance grid, elevation and gradient interpolated.

    Even spacing is a contract, not a detail. A profile chart draws one mark per
    point and sizes every mark from the *smallest* gap in the series, so an
    unevenly sampled payload draws a hairline comb — and a payload with two
    samples at the same distance sizes every mark to zero and draws nothing at
    all. Both are what the athlete sees: an empty chart with correct axes.

    Snapping each grid target to the last source point at or below it (what this
    did before) produces exactly that. It is safe only while the source is dense
    everywhere: a GPX from a route planner is not — long straights carry a point
    per kilometre while junctions carry one every few metres — so wherever the
    track was sparser than the grid, consecutive targets snapped to the *same*
    point and the payload carried repeated distances.

    Interpolating between the two bracketing samples instead is both uniform and
    honest: between two real samples the chart draws a straight line either way,
    so the reconstruction is the one the athlete would have seen, on a grid the
    chart can actually mark. The grid never carries more samples than the source
    did — a sparse course stays a sparse course, at its own resolution — and its
    ends are the source's own first and last points, exactly.
    """
    n = min(CHART_PROFILE_MAX_POINTS, len(points))
    if n < 2:
        return list(points)
    start = points[0].distance_m
    total = points[-1].distance_m - start
    if total <= 0:
        # No distance to spread a grid over (a stationary or single-place
        # track). Nothing to resample onto; hand back what there is.
        return list(points[:n])

    out: list[ProfilePoint] = []
    # `j` walks forward with the targets — the grid is monotonic, so the search
    # for each bracketing pair resumes where the last one ended.
    j = 0
    last = len(points) - 1
    for i in range(n):
        target = start + total * i / (n - 1)
        while j + 1 < last and points[j + 1].distance_m <= target:
            j += 1
        low, high = points[j], points[j + 1]
        span = high.distance_m - low.distance_m
        frac = (target - low.distance_m) / span if span > 0 else 0.0
        frac = min(max(frac, 0.0), 1.0)
        out.append(
            ProfilePoint(
                distance_m=target,
                elevation_m=low.elevation_m + frac * (high.elevation_m - low.elevation_m),
                gradient=low.gradient + frac * (high.gradient - low.gradient),
            )
        )
    return out


def analyze_course(
    profile: CourseProfile,
    rider: RiderParams,
    bike: BikeParams,
    target_time_s: float | None = None,
    target_power_w: float | None = None,
) -> CourseAnalysis:
    """The full analysis of a course profile — coordinate-free by construction.

    Takes the profile rather than a track on purpose: the one conversion in
    :func:`course_profile` is where coordinates end, and nothing reachable
    from here can carry one.

    The two targets are alternatives — a ride is paced *to a finish time* or
    *to a number of watts*, never to both — and the API refuses a request
    carrying both. Should one arrive anyway, the power target wins: it is the
    one the model can always honour.
    """
    elevations = [p.elevation_m for p in profile.points]
    gain = geo.elevation_gain_m(elevations, smoothing=1)
    loss = geo.elevation_gain_m(list(reversed(elevations)), smoothing=1)

    segments = segment_by_gradient(profile)
    if target_power_w is not None:
        pacing = solve_target_power(segments, rider, bike, target_power_w)
    elif target_time_s is not None:
        pacing = solve_target_time(segments, rider, bike, target_time_s)
    else:
        pacing = default_pacing(segments, rider, bike)

    return CourseAnalysis(
        total_distance_m=profile.total_distance_m,
        elevation_gain_m=gain,
        elevation_loss_m=loss,
        min_elevation_m=min(elevations),
        max_elevation_m=max(elevations),
        profile=_resample_profile(profile.points),
        segments=segments,
        pacing=pacing,
    )
