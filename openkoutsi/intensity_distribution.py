"""Three-band intensity distribution over a training block (issue #38).

Weekly time-in-zone answers "what did I do last week". This module answers the
question periodization is organised around: over a block, did the training come
out *polarized*, *pyramidal* or *threshold-heavy*?

That question is posed against a three-zone model — below LT1, between LT1 and
LT2, above LT2 — not the Coggan zones openkoutsi stores. Since zone lists are
now fixed at seven power / five HR zones (see ``zones.POWER_ZONE_COUNT``), the
mapping down to three bands is positional rather than inferred: LT1 sits at the
Z2/Z3 boundary and LT2 at the Z4/Z5 boundary, both taken from the athlete's own
zones, so no "LT1 is 0.75 × FTP" constant is needed anywhere.

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


_ZONE_NUMBER = re.compile(r"\d+")


def _zone_sort_key(name: str) -> tuple[int, int, str]:
    """Order snapshot keys by zone number.

    Snapshots are keyed by zone *name*, and those names vary by where the zones
    came from: ``"Z1 Recovery"`` from the app's defaults, bare ``"Z1"`` from
    provider sync. Sorting on the leading number recovers the ascending order
    the positional mapping depends on. Names with no number sort last, so an
    unrecognisable snapshot degrades predictably instead of scrambling.
    """
    match = _ZONE_NUMBER.search(name)
    if match is None:
        return (1, 0, name)
    return (0, int(match.group()), name)


def sort_zone_names(names) -> list[str]:
    """Snapshot keys in ascending zone order."""
    return sorted(names, key=_zone_sort_key)


def zone_number(name: str, fallback: int) -> int:
    """The 1-based zone number in ``name``, or ``fallback`` when it has none."""
    match = _ZONE_NUMBER.search(name)
    return int(match.group()) if match else fallback


def bands_from_zone_times(zone_times: Mapping | None, basis: str) -> dict[int, int]:
    """Seconds per band from one activity's frozen ``zone_times`` snapshot.

    Returns a band → seconds dict with all three bands present (zeroed when the
    snapshot has nothing for this basis).

    Each zone's position comes from the number in its name rather than from its
    place among the keys present, because a snapshot only carries the zones the
    ride actually touched. An easy ride that never left Z1–Z3 stores three keys;
    reading those as a three-zone model would promote Z3 into the top band and
    report a recovery spin as high intensity.
    """
    if basis not in _CANONICAL_COUNTS:
        raise ValueError(f"unknown basis: {basis!r}")

    totals = {band: 0 for band in BANDS}
    times = (zone_times or {}).get(basis) or {}
    if not times:
        return totals

    indexed: list[tuple[int, int]] = []  # (zone index, seconds)
    for position, name in enumerate(sort_zone_names(times)):
        index = max(zone_number(name, position + 1) - 1, 0)
        indexed.append((index, int(times.get(name) or 0)))

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

    Rule 2 is deliberately ahead of rule 3: a block with a lot of band 2 *and*
    a little more band 3 is the grey-zone grind this feature exists to expose,
    and calling it polarized would bury exactly that. The consequence is that a
    degenerate all-band-3 block comes out ``polarized``; no label describes that
    case well, and it is pinned by a test so the behaviour can't drift silently.

    These cut-offs are conventions rather than physical constants.
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
