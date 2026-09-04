"""Three-band intensity distribution over a training block (issue #38).

Weekly time-in-zone answers "what did I do last week". This module answers the
question periodization is organised around: over a block, did the training come
out *polarized*, *pyramidal* or *threshold-heavy*?

The question is posed against a three-zone model — below LT1, LT1–LT2, above
LT2 — not the Coggan zones openkoutsi stores. Zone lists are fixed at seven
power / five HR zones (``zones.POWER_ZONE_COUNT``), so the mapping is positional:
LT1 at the Z2/Z3 boundary, LT2 at Z4/Z5, both from the athlete's own zones, so no
"LT1 is 0.75 × FTP" constant is needed.

Everything here is pure: no DB, no I/O, percentages in and shapes out.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

from openkoutsi.categorization import WorkoutCategory
from openkoutsi.zones import HR_ZONE_COUNT, POWER_ZONE_COUNT

# The three bands. 1 = below LT1, 2 = between LT1 and LT2, 3 = above LT2.
BAND_LOW = 1
BAND_MODERATE = 2
BAND_HIGH = 3
BANDS = (BAND_LOW, BAND_MODERATE, BAND_HIGH)

BASES = ("power", "hr")

# Where the band boundaries fall in a canonical zone list, as the index of the
# first zone of band 2 and of band 3. Both models split the same way — the two
# easiest zones, then the two around threshold, then everything above:
#   power (7): Z1 Z2 | Z3 Z4 | Z5 Z6 Z7
#   HR    (5): Z1 Z2 | Z3 Z4 | Z5
_CANONICAL_SPLITS = (2, 4)
_CANONICAL_COUNTS = {"power": POWER_ZONE_COUNT, "hr": HR_ZONE_COUNT}

# Shapes. These are conventions, not physical constants — see ``classify``.
POLARIZED = "polarized"
PYRAMIDAL = "pyramidal"
THRESHOLD = "threshold"
PREDOMINANTLY_LOW = "predominantly_low"
SHAPES = (POLARIZED, PYRAMIDAL, THRESHOLD, PREDOMINANTLY_LOW)

# A block with almost nothing above LT1 is not meaningfully any of the three
# shapes, so it gets its own label rather than being forced into one.
LOW_INTENSITY_GUARD_PCT = 10.0
# Band 2 at or above this share reads as threshold work even when band 1 is
# still the largest band.
THRESHOLD_BAND_PCT = 35.0


def _splits(zone_count: int, basis: str) -> tuple[int, int]:
    """Band boundaries for a zone list of ``zone_count`` entries.

    Canonical lists use the fixed table. Anything else is a snapshot frozen
    before the zone count was pinned down, so the boundaries are placed at the
    same *proportions* of the list instead — a best effort that keeps old
    history readable rather than dropping it.
    """
    canonical = _CANONICAL_COUNTS[basis]
    if zone_count == canonical:
        return _CANONICAL_SPLITS

    lo, hi = _CANONICAL_SPLITS
    first_moderate = min(zone_count, max(1, round(zone_count * lo / canonical)))
    first_high = min(zone_count, max(first_moderate, round(zone_count * hi / canonical)))
    return first_moderate, first_high


def band_for_zone_index(index: int, zone_count: int, basis: str) -> int:
    """Band that the ``index``-th zone (0-based, ascending) belongs to."""
    if basis not in _CANONICAL_COUNTS:
        raise ValueError(f"unknown basis: {basis!r}")

    first_moderate, first_high = _splits(zone_count, basis)
    if index < first_moderate:
        return BAND_LOW
    if index < first_high:
        return BAND_MODERATE
    return BAND_HIGH


# Anchored deliberately. An unanchored ``\d+`` takes the first digit run
# *anywhere* in the name, and zone names were free-form before the model was
# fixed, so it mis-parsed real ones: ``VO2max`` read as zone 2 and filed hard
# work below LT1, and ``Sweet Spot 88-94%`` read as zone 88, which rescaled the
# band boundaries for every other zone in the same snapshot and inverted the
# distribution. Anchoring accepts ``Z1 Recovery``, ``Zone 1``, ``1 Recovery``
# and ``Z1``, and refuses the rest outright rather than guessing at them.
_ZONE_NUMBER = re.compile(r"^\s*(?:zone\s*|z\s*)?(\d+)", re.IGNORECASE)

# A parsed number this far above the canonical count is not a zone number —
# it's a percentage or a wattage that happened to lead the name. Refuse it
# rather than letting one value redefine the model for the whole snapshot.
_MAX_ZONE_NUMBER_FACTOR = 2


def zone_number(name: str) -> int | None:
    """The 1-based zone number leading ``name``, or ``None`` if it has none."""
    match = _ZONE_NUMBER.match(name)
    if match is None:
        return None
    return int(match.group(1))


def _zone_sort_key(name: str) -> tuple[int, int, str]:
    """Order snapshot keys by zone number.

    Snapshots are keyed by zone *name*, and those names vary by where the zones
    came from: ``"Z1 Recovery"`` from the app's defaults, bare ``"Z1"`` from
    provider sync. Names with no parseable number sort last; nothing is derived
    from their position, they only need a deterministic order.
    """
    number = zone_number(name)
    if number is None:
        return (1, 0, name)
    return (0, number, name)


def sort_zone_names(names) -> list[str]:
    """Snapshot keys in ascending zone order."""
    return sorted(names, key=_zone_sort_key)


def bands_from_zone_times(zone_times: Mapping | None, basis: str) -> dict[int, int]:
    """Seconds per band from one activity's frozen ``zone_times`` snapshot.

    Returns a band → seconds dict with all three bands present (zeroed when the
    snapshot has nothing for this basis).

    Each zone's position comes from the number in its name, not its place among
    the keys present: a snapshot carries only the zones the ride touched, so
    reading an easy ride's three keys as a three-zone model would report a
    recovery spin as high intensity.

    A snapshot carrying any name the mapping can't place returns **all zeros**,
    dropping the activity out of ``activities_used`` as honest coverage.
    Snapshots frozen before zone names were normalised can hold anything the
    athlete typed, and there is no order to recover from ``Recovery, Endurance,
    Threshold``.
    """
    if basis not in _CANONICAL_COUNTS:
        raise ValueError(f"unknown basis: {basis!r}")

    totals = {band: 0 for band in BANDS}
    times = (zone_times or {}).get(basis) or {}
    if not times:
        return totals

    limit = _CANONICAL_COUNTS[basis] * _MAX_ZONE_NUMBER_FACTOR
    indexed: list[tuple[int, int]] = []  # (zone index, seconds)
    for name in sort_zone_names(times):
        number = zone_number(name)
        if number is None or number < 1 or number > limit:
            return {band: 0 for band in BANDS}
        indexed.append((number - 1, int(times.get(name) or 0)))

    # Assume the canonical model unless the snapshot proves it had more zones.
    zone_count = max(_CANONICAL_COUNTS[basis], max(i for i, _ in indexed) + 1)
    for index, seconds in indexed:
        totals[band_for_zone_index(index, zone_count, basis)] += seconds
    return totals


# Session-goal band per workout category. Tempo sits in band 2 alongside
# threshold, matching where Z3 Tempo lands in the time-in-zone mapping — the
# two methods disagreeing about the same ride would be worse than either
# convention on its own. The non-cycling categories carry no cycling intensity,
# so they are excluded outright rather than forced into a band.
_CATEGORY_BANDS: dict[WorkoutCategory, int | None] = {
    WorkoutCategory.recovery: BAND_LOW,
    WorkoutCategory.endurance: BAND_LOW,
    WorkoutCategory.tempo: BAND_MODERATE,
    WorkoutCategory.threshold: BAND_MODERATE,
    WorkoutCategory.vo2max: BAND_HIGH,
    WorkoutCategory.anaerobic: BAND_HIGH,
    WorkoutCategory.sprint: BAND_HIGH,
    WorkoutCategory.strength: None,
    WorkoutCategory.yoga: None,
    WorkoutCategory.cross_training: None,
}


def band_for_category(category: WorkoutCategory | str | None) -> int | None:
    """Band for a session-goal count, or ``None`` when the session is excluded.

    ``None`` covers an unset category, a non-cycling one, and anything
    unrecognised. Callers count those against coverage rather than silently
    dropping them.
    """
    if category is None:
        return None
    if not isinstance(category, WorkoutCategory):
        try:
            category = WorkoutCategory(category)
        except ValueError:
            return None
    return _CATEGORY_BANDS.get(category)


def band_percentages(totals: Mapping[int, float]) -> dict[int, float]:
    """Band totals → percentages of their sum. All-zero input gives all zeros."""
    total = sum(totals.get(band, 0) for band in BANDS)
    if total <= 0:
        return {band: 0.0 for band in BANDS}
    return {band: totals.get(band, 0) * 100.0 / total for band in BANDS}


def classify(low_pct: float, moderate_pct: float, high_pct: float) -> str | None:
    """Name the shape of a distribution, or ``None`` for an empty window.

    The rule *order* is the specification — the rules overlap, and which one
    wins is the decision:

    1. almost nothing above LT1 → ``predominantly_low``
    2. band 2 is large, or is the biggest band → ``threshold``
    3. more above LT2 than between the thresholds → ``polarized``
    4. otherwise → ``pyramidal``

    Rule 2 is deliberately ahead of rule 3: a block with a lot of band 2 and a
    little more band 3 is the grey-zone grind this feature exposes, which
    "polarized" would bury. The consequence is that a degenerate all-band-3 block
    comes out ``polarized``, which a test pins so it can't drift silently.

    These cut-offs are conventions, not physical constants.
    """
    if low_pct + moderate_pct + high_pct <= 0:
        return None

    if moderate_pct + high_pct < LOW_INTENSITY_GUARD_PCT:
        return PREDOMINANTLY_LOW

    if moderate_pct >= THRESHOLD_BAND_PCT or (
        moderate_pct >= low_pct and moderate_pct >= high_pct
    ):
        return THRESHOLD

    if high_pct > moderate_pct:
        return POLARIZED

    return PYRAMIDAL
