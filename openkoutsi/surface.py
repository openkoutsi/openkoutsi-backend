"""Road surface classification for course recon (issue #56, Stage 2).

Stage 1 solves every course as dry pavement. This module is what replaces that
assumption with a class per metre of road, matched against OpenStreetMap by an
optional routing sidecar the self-hoster runs themselves.

Everything here is pure — plain strings and floats in and out, no I/O. The
matcher client lives in the backend (``backend/app/services/surface_matcher``);
this module owns the vocabulary, the physics constants, and the judgement about
what a match is actually worth.

Three things it is careful about, in the order they matter:

1. **Confidence is not decoration.** OSM surface coverage is genuinely uneven —
   dense across Germany and the Netherlands, thin across rural North America —
   and a guess presented beside a fact at equal weight is worse than no answer.
   See :func:`confidence_for`, which is exact rather than heuristic.
2. **A short sector is not automatically noise.** A rider cannot expect to roll
   through 40 km of asphalt if 130 m of it is mud and rocks. :func:`dissolve_runs`
   decides on *severity*, not length alone, so a snap artefact disappears and a
   real sector survives.
3. **An unknown value classifies as unknown**, never as a default that looks
   confident.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


# ── the class set ─────────────────────────────────────────────────────────────

# Deliberately 1:1 with the routing engine's own vocabulary (plus UNKNOWN)
# rather than a coarser grouping of our own. Valhalla has already normalised
# OSM's free-form `surface` values into these eight; inventing a second,
# lossier grouping on top would be a judgement we have no data to support.
ASPHALT = "asphalt"
PAVED = "paved"
COBBLES = "cobbles"
COMPACTED = "compacted"
GRAVEL = "gravel"
DIRT = "dirt"
GRASS = "grass"
UNKNOWN = "unknown"

SURFACE_CLASSES = (
    ASPHALT,
    PAVED,
    COBBLES,
    COMPACTED,
    GRAVEL,
    DIRT,
    GRASS,
    UNKNOWN,
)

#: Matcher value → our class. `impassable` maps to UNKNOWN rather than to a
#: class of its own: on a course the athlete intends to ride it is far more
#: likely to be a bad snap than a real impassable road, and inventing a scary
#: label from a probable match error is its own kind of dishonesty.
_FROM_MATCHER = {
    "paved_smooth": ASPHALT,
    "paved": PAVED,
    "paved_rough": COBBLES,
    "compacted": COMPACTED,
    "gravel": GRAVEL,
    "dirt": DIRT,
    "path": GRASS,
    "impassable": UNKNOWN,
}

#: The value the routing engine returns for a way carrying no surface
#: information at all. Load-bearing — see :func:`confidence_for`.
UNTAGGED_MATCHER_VALUE = "paved_smooth"

CONFIRMED = "confirmed"
INFERRED = "inferred"
CONFIDENCES = (CONFIRMED, INFERRED)


def normalise(raw: str | None) -> str:
    """Matcher surface value → one of :data:`SURFACE_CLASSES`.

    A value we do not recognise — a new enum member from a future engine
    version, a typo, ``None`` from an unmatched point — becomes
    :data:`UNKNOWN`. It must never fall through to a default that reads as
    confident: "we could not tell" and "smooth tarmac" are different answers
    and the athlete is entitled to know which one they got.
    """
    if raw is None:
        return UNKNOWN
    return _FROM_MATCHER.get(raw.strip().lower(), UNKNOWN)


def confidence_for(raw: str | None) -> str:
    """Whether the matched surface rests on an explicit OSM tag.

    This is exact, not a heuristic, and it follows from how the routing engine
    builds its graph. ``OSMWay`` stores the surface in a three-bit field zeroed
    on construction, and ``kPavedSmooth`` is enumerator 0 — so a way that
    carries no surface information at all comes back as ``paved_smooth``. Every
    *other* value is reachable only from an explicit tag: ``surface=*``, or,
    where that is absent, ``tracktype=grade1..5``, ``smoothness=*``, or
    ``sac_scale`` / ``mtb:*``.

    So:

    - ``paved_smooth`` → :data:`INFERRED`. The way is either explicitly paved
      or simply untagged, and **we cannot tell the two apart**, so we report
      the weaker claim.
    - anything else → :data:`CONFIRMED`. Somebody tagged this road.
    - unrecognised or absent → :data:`INFERRED`.

    Note what this deliberately does *not* claim. ``inferred`` means "openkoutsi
    could not confirm a surface tag here", **not** "this road is untagged" — a
    genuinely tagged asphalt road reads as inferred too. That under-claims, and
    under-claiming is the safe direction; the UI and the docs must use the same
    wording rather than the shorter, wronger one.
    """
    if raw is None:
        return INFERRED
    # A class we could not identify can never be confirmed — `impassable`
    # normalises to UNKNOWN, so it must not report as a fact merely because it
    # is a value the engine recognises. Deriving from the class rather than
    # from the raw string keeps "unknown implies inferred" true by
    # construction rather than by two rules agreeing.
    if normalise(raw) == UNKNOWN:
        return INFERRED
    return INFERRED if raw.strip().lower() == UNTAGGED_MATCHER_VALUE else CONFIRMED


# ── severity ──────────────────────────────────────────────────────────────────

# The classes on an ordered roughness scale — the same ordering the Crr table
# below induces, so "rougher" and "slower" cannot drift apart.
_SEVERITY_RANK = {
    ASPHALT: 0,
    PAVED: 1,
    COMPACTED: 2,
    COBBLES: 3,
    GRAVEL: 4,
    DIRT: 5,
    GRASS: 6,
}

#: A run at least this rough is worth naming to the rider in prose, whatever
#: its length. Compacted hardpack is where a road stops being a road.
ROUGH_SECTOR_MIN_RANK = _SEVERITY_RANK[COMPACTED]


def severity_rank(surface: str | None) -> int:
    """Where a class sits on the roughness scale. UNKNOWN ranks as smooth."""
    return _SEVERITY_RANK.get(surface or UNKNOWN, 0)


def severity_delta(a: str | None, b: str | None) -> int:
    """How big a change ``a`` → ``b`` is, in ranks.

    Zero when either side is :data:`UNKNOWN`: we have no basis for calling a
    surface we could not identify a *change*, and an unknown run must not be
    able to survive the dissolve by pretending to be dramatic.
    """
    if a is None or b is None or a == UNKNOWN or b == UNKNOWN:
        return 0
    return abs(severity_rank(a) - severity_rank(b))


# ── rolling resistance ────────────────────────────────────────────────────────

# Base Crr per surface at a mid-width (~40 mm) tyre, from drum and field
# measurements (bicyclerollingresistance.com; Wilson, *Bicycling Science*).
# Coarse on purpose: the spread between two gravel roads dwarfs the precision
# of any single figure, and a plan that implied otherwise would be pretending.
_CRR_BASE = {
    PAVED: 0.0080,
    COMPACTED: 0.0100,
    COBBLES: 0.0150,
    GRAVEL: 0.0180,
    DIRT: 0.0220,
    GRASS: 0.0350,
}

def crr_for_tyre_width(width_mm: int | None) -> float:
    """Rolling-resistance coefficient on pavement, by tyre width.

    Measured Crr for road tyres spans roughly 0.003–0.007
    (bicyclerollingresistance.com drum tests). Width is a workable proxy: the
    26–32 mm range rolls best at sensible pressures, narrower gives a little
    back on real road surfaces, and wider means heavier casings. Stage 1
    assumed pavement throughout; :func:`crr_for` is the Stage 2 (#56)
    generalisation, and delegates here for the paved classes so the two cannot
    drift apart.
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


# Off pavement the width relationship **inverts**: on tarmac a wide tyre costs
# you casing losses, but on a loose surface it floats over what a narrow one
# cuts into. Relative to the ~40 mm reference the base table is quoted at.
def _width_factor(width_mm: int | None) -> float:
    if width_mm is None:
        # No information: treat as a mid-width tyre rather than assuming the
        # athlete is on something heroic in either direction.
        return 1.0
    if width_mm <= 25:
        return 1.45
    if width_mm <= 32:
        return 1.25
    if width_mm <= 45:
        return 1.00
    return 0.88


def crr_for(surface: str | None, width_mm: int | None) -> float:
    """Rolling-resistance coefficient for a surface class and tyre width.

    On :data:`ASPHALT`, and wherever the surface is unknown or absent, this
    returns exactly what :func:`openkoutsi.course.crr_for_tyre_width` returns.
    That continuity is deliberate and is asserted in the tests: a course with
    no surface data, or one matched as paved throughout, must produce the same
    numbers Stage 1 produced, so enabling the sidecar never silently restates
    an existing plan.
    """
    if surface is None or surface == ASPHALT or surface == UNKNOWN:
        return crr_for_tyre_width(width_mm)
    base = _CRR_BASE.get(surface)
    if base is None:
        return crr_for_tyre_width(width_mm)
    return base * _width_factor(width_mm)


# ── the per-point series ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class SurfacePoint:
    """One matched track point, before any dissolving."""

    surface: str  # normalised class as matched
    confidence: str
    raw: str | None  # exactly what the matcher said, preserved not discarded


@dataclass(frozen=True)
class SurfaceRun:
    """A maximal stretch of one class, after dissolving."""

    start_distance_m: float
    end_distance_m: float
    surface: str
    confidence: str
    raw: str | None
    #: Rank step against the run before it. Positive means the road got worse,
    #: which is the part a rider needs warning about.
    severity_step: int

    @property
    def length_m(self) -> float:
        return self.end_distance_m - self.start_distance_m


@dataclass(frozen=True)
class RoughSector:
    """A stretch worth naming in prose — the sibling of a key climb."""

    start_distance_m: float
    length_m: float
    surface: str
    confidence: str
    severity_step: int


# ── dissolving ────────────────────────────────────────────────────────────────

# The floor for a *low-severity* change — a class step of one rank, which is
# usually the matcher snapping to the service road beside the road you are
# actually on. Scaled down as severity rises: see `_required_run_length_m`.
MIN_SURFACE_RUN_M = 150.0

# The one unconditional floor. Below a couple of match samples the match itself
# is not trustworthy at any severity, and a two-point "mud sector" is far more
# likely to be a bad snap than a real one.
HARD_MIN_RUN_M = 40.0

# Severity only buys a short run its life when the run is *isolated* — when
# the stretches either side of it are themselves substantial. A short run
# surrounded by other short runs is a match storm, and a storm of 50 m
# alternations is not twenty gravel sectors however dramatic each looks on its
# own; nothing in it is trustworthy enough to pace or warn on. Without this,
# the severity rule that protects a real 130 m mud sector would also let
# snapping between a road and its parallel cycleway shatter the route.
ISOLATION_MIN_NEIGHBOUR_M = 2 * HARD_MIN_RUN_M

#: A segment is reported as confirmed only when this share of its length was
#: matched to the segment's own class *and* confirmed there.
SEGMENT_CONFIRMED_FRACTION = 0.6


def _required_run_length_m(step: int) -> float:
    """How long a run has to be to survive, given how severe the change is.

    A one-rank step needs the full :data:`MIN_SURFACE_RUN_M`; a jump of several
    ranks survives at the hard floor. This is the whole point of the pass: a
    40 m stretch of paving stones inside asphalt is noise and should vanish,
    while 130 m of dirt inside 40 km of asphalt is the most important thing on
    the course and must not.
    """
    return max(HARD_MIN_RUN_M, MIN_SURFACE_RUN_M / (1 + max(0, step)))


def dissolve_runs(
    points: Sequence[SurfacePoint],
    distances_m: Sequence[float],
) -> list[str]:
    """Rewrite match noise out of the per-point class series.

    Returns one class per input point — the *dissolved* class, which for a
    surviving run is simply what was matched. Callers keep ``points`` alongside
    the result: the original class is what :func:`run_confidence` measures
    against, so a class that exists only because a blip was dissolved cannot be
    laundered into ``confirmed``.

    A run is a maximal stretch of one class, so every run considered here is
    bounded by *different* classes — merging across a surface change is the
    job, not a side effect. When a run does dissolve, the longer neighbour
    wins, ties going to the preceding run (matching the gradient dissolve in
    :func:`openkoutsi.course.segment_by_gradient`).
    """
    if len(points) != len(distances_m):
        raise ValueError("points and distances_m must be the same length")
    if not points:
        return []

    runs = _initial_runs(points)
    last = len(distances_m) - 1

    def _length(run: list) -> float:
        return max(0.0, distances_m[min(run[1], last)] - distances_m[run[0]])

    # Same linked-list traversal as the gradient dissolve, and for the same
    # reason: restarting the scan after every merge is quadratic in the run
    # count, worst on exactly the terrain the dissolve exists for.
    n = len(runs)
    prev = list(range(-1, n - 1))
    nxt = list(range(1, n + 1))
    if nxt:
        nxt[-1] = -1
    dead = [False] * n

    def _absorb(target: int, other: int) -> None:
        """Extend ``target`` over ``other`` and unlink ``other``."""
        runs[target][0] = min(runs[target][0], runs[other][0])
        runs[target][1] = max(runs[target][1], runs[other][1])
        p, q = prev[other], nxt[other]
        if p != -1:
            nxt[p] = q
        if q != -1:
            prev[q] = p
        dead[other] = True

    i = 0 if n else -1
    while i != -1:
        run = runs[i]
        p, q = prev[i], nxt[i]
        if p == -1 and q == -1:
            break  # the only run left: nothing to dissolve into

        # Severity is measured against the *nearer* neighbour, not the
        # further one, because what earns a short run its life is standing
        # out — not merely being far from one side. A stretch of paving
        # stones between asphalt and gravel sits between its neighbours in
        # roughness: it is a transition, and a 40 m one is a snap artefact.
        # A stretch of dirt between two stretches of asphalt is an outlier
        # from both, and that is the case worth keeping.
        neighbours = [k for k in (p, q) if k != -1]
        deltas = [severity_delta(run[2], runs[k][2]) for k in neighbours]
        isolated = all(
            _length(runs[k]) >= ISOLATION_MIN_NEIGHBOUR_M for k in neighbours
        )
        step = min(deltas) if (deltas and isolated) else 0
        if _length(run) >= _required_run_length_m(step):
            i = q
            continue

        if p == -1:
            target = q
        elif q == -1:
            target = p
        else:
            # The longer neighbour wins; ties go to the preceding run.
            target = p if _length(runs[p]) >= _length(runs[q]) else q
        _absorb(target, i)

        # Removing a run can leave two same-class runs adjacent (A B A, with B
        # dissolved). Coalesce them so the series stays run-length encoded.
        after = nxt[target]
        if after != -1 and runs[after][2] == runs[target][2]:
            _absorb(target, after)
        before = prev[target]
        if before != -1 and runs[before][2] == runs[target][2]:
            _absorb(before, target)
            target = before

        # Resume at the merged run rather than restarting: it may now be long
        # enough, or may itself need dissolving, and either way everything
        # before it was already checked in this same traversal.
        i = target

    out: list[str | None] = [None] * len(points)
    for k in range(n):
        if dead[k]:
            continue
        for j in range(runs[k][0], min(runs[k][1], len(points))):
            out[j] = runs[k][2]
    return [out[j] or points[j].surface for j in range(len(points))]


def _initial_runs(points: Sequence[SurfacePoint]) -> list[list]:
    """Maximal same-class stretches as ``[start, end_exclusive, surface]``."""
    runs: list[list] = []
    for i, point in enumerate(points):
        if runs and runs[-1][2] == point.surface:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1, point.surface])
    return runs


def run_confidence(
    points: Sequence[SurfacePoint],
    lengths_m: Sequence[float],
    surface: str,
) -> str:
    """Confidence for a stretch, from the points underneath it.

    Confirmed only when the share of length whose *originally matched* class
    equals ``surface`` **and** was itself confirmed clears
    :data:`SEGMENT_CONFIRMED_FRACTION`. Two things fall out, both of which are
    the honest answer:

    - a stretch whose class exists only because a blip was dissolved away reads
      inferred, because those points were matched as something else;
    - a genuinely mixed stretch — half confirmed gravel, half road that
      defaulted to paved — reads inferred too.
    """
    total = sum(lengths_m)
    if total <= 0:
        return INFERRED
    agreeing = sum(
        length
        for point, length in zip(points, lengths_m)
        if point.surface == surface and point.confidence == CONFIRMED
    )
    return CONFIRMED if agreeing / total >= SEGMENT_CONFIRMED_FRACTION else INFERRED


def build_runs(
    points: Sequence[SurfacePoint],
    distances_m: Sequence[float],
) -> list[SurfaceRun]:
    """The dissolved class series as run-length-encoded runs.

    This is the honest-resolution record of the course's surface, and it is
    stored separately from both the chart profile and the segment table
    precisely because those two have minimum resolutions and this does not. A
    130 m mud sector survives here even when the pacing table quite reasonably
    folds it into a longer row.
    """
    if not points:
        return []
    dissolved = dissolve_runs(points, distances_m)

    bounds: list[list[int]] = []
    for i, surface in enumerate(dissolved):
        if bounds and dissolved[bounds[-1][0]] == surface:
            bounds[-1][1] = i + 1
        else:
            bounds.append([i, i + 1])

    runs: list[SurfaceRun] = []
    last = len(distances_m) - 1
    for start, end in bounds:
        end_index = min(end, last)
        span = [
            distances_m[min(j + 1, last)] - distances_m[j] for j in range(start, end_index)
        ]
        surface = dissolved[start]
        member = points[start : max(end_index, start + 1)]
        confidence = run_confidence(member, span or [1.0], surface)
        previous = runs[-1].surface if runs else None
        runs.append(
            SurfaceRun(
                start_distance_m=distances_m[start],
                end_distance_m=distances_m[end_index],
                surface=surface,
                confidence=confidence,
                raw=points[start].raw,
                severity_step=(
                    severity_rank(surface) - severity_rank(previous) if previous else 0
                ),
            )
        )
    return runs


def rough_sectors(runs: Sequence[SurfaceRun]) -> list[RoughSector]:
    """Stretches worth naming to the rider, in the manner of a key climb.

    Every run at least :data:`ROUGH_SECTOR_MIN_RANK` rough, **including ones
    too short to earn their own pacing segment** — which is the entire reason
    this exists. A colour on a chart is missable; "130 m of mud from km 41.2"
    read out in the plan is not, and a rider who expected 40 km of tarmac needs
    that sentence before the day rather than after it.
    """
    return [
        RoughSector(
            start_distance_m=run.start_distance_m,
            length_m=run.length_m,
            surface=run.surface,
            confidence=run.confidence,
            severity_step=run.severity_step,
        )
        for run in runs
        if severity_rank(run.surface) >= ROUGH_SECTOR_MIN_RANK and run.length_m > 0
    ]


def ribbon_json(runs: Sequence[SurfaceRun]) -> list:
    """The runs in the compact form stored on the course row."""
    return [
        [
            round(run.start_distance_m, 1),
            round(run.end_distance_m, 1),
            run.surface,
            run.confidence,
            run.severity_step,
        ]
        for run in runs
    ]


def rough_sector_json(ribbon: Sequence | None) -> list:
    """The rough stretches of a stored ribbon, in the shape the API serves.

    Derived from the ribbon rather than stored beside it: the ribbon already
    says what the surface is at full resolution, and a second copy is a second
    thing that can disagree with it.
    """
    out: list = []
    for entry in ribbon or []:
        start, end, klass, confidence, step = (list(entry) + [0])[:5]
        if end > start and severity_rank(klass) >= ROUGH_SECTOR_MIN_RANK:
            out.append([start, end - start, klass, confidence, step])
    return out


def points_from_json(stored: Sequence | None) -> list[SurfacePoint] | None:
    """Rebuild the per-point series stored beside a course's track.

    Storage keeps only ``[raw, confidence]`` per point: the class and every
    dissolving decision are re-derived here, so tuning a threshold later
    re-reads correctly from what is already on disk instead of needing every
    course re-matched.
    """
    if not stored:
        return None
    points: list[SurfacePoint] = []
    for entry in stored:
        raw = entry[0] if entry else None
        confidence = entry[1] if entry and len(entry) > 1 else None
        points.append(
            SurfacePoint(
                surface=normalise(raw),
                confidence=confidence if confidence in CONFIDENCES else confidence_for(raw),
                raw=raw,
            )
        )
    return points
