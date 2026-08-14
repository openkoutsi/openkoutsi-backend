from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from . import streams as stream_utils

# The canonical zone models. Zone lists used to be arbitrary-length, which made
# anything built on top of them (see ``intensity_distribution``) guess at what a
# given zone meant. They are now fixed: seven power zones (Coggan) and five HR
# zones, both running Z1 (recovery) → Zn (maximal). The API rejects other
# lengths; snapshots frozen before that still carry whatever count they were
# computed with, so readers must degrade rather than assume.
POWER_ZONE_COUNT = 7  # Z1 Recovery … Z7 Neuromuscular
HR_ZONE_COUNT = 5  # Z1 Recovery … Z5 VO2max

# Canonical zone names. The API owns these: names are normalised on write, so
# the athlete's text is a label rather than data. That matters because the
# frozen ``zone_times`` snapshot is keyed by name, and anything reading a
# snapshot has to recover the zone's position from it — an invariant that
# cannot be left to whatever an athlete typed into a free-text field.
POWER_ZONE_NAMES = (
    "Z1 Recovery",
    "Z2 Endurance",
    "Z3 Tempo",
    "Z4 Threshold",
    "Z5 VO2max",
    "Z6 Anaerobic",
    "Z7 Neuromuscular",
)

HR_ZONE_NAMES = (
    "Z1 Recovery",
    "Z2 Endurance",
    "Z3 Tempo",
    "Z4 Threshold",
    "Z5 VO2max",
)


def time_in_zones(
    samples: Iterable[float | None], zone_defs: Sequence[dict]
) -> dict[str, int]:
    """Accumulate time spent in each zone from a per-second sample stream.

    ``samples`` is a 1 Hz stream (one value per second), so each sample counts
    as one second. ``zone_defs`` is the athlete's zone list — ``[{"low", "high",
    "name"}, ...]``. Returns ``{zone_name: seconds}``. Values below Z1 / above
    the last zone are clamped into the nearest zone by ``Zones.getZone``.

    Gaps (``None``/``NaN``, see ``openkoutsi.streams``) are seconds the sensor
    recorded nothing, and are counted as time in no zone at all rather than
    apportioned to one. They must be dropped *before* the cast below: ``NaN``
    casts to ``INT64_MIN``, which then clamps into Z1 and would book a
    ten-minute strap dropout as ten minutes of recovery riding.
    """
    zones = Zones(*[(z["low"], z["high"]) for z in zone_defs])
    # ``.astype`` truncates toward zero, matching the ``int(v)`` this used to do
    # per sample before handing the value to ``getZone``.
    values = stream_utils.present(stream_utils.as_array(samples)).astype(np.int64)
    counts = np.bincount(zones.zoneIndices(values), minlength=len(zone_defs))

    out: dict[str, int] = {}
    for i, seconds in enumerate(counts):
        if seconds:
            name = zone_defs[i].get("name", f"Z{i + 1}")
            out[name] = out.get(name, 0) + int(seconds)
    return out


class Zones:
    def __init__(
        self,
        *_zones: tuple[int, int]
    ) -> None:
        self.zones = []
        for z in _zones:
            self.zones.append(z)

        self.validate()

    def zoneName(self, i) -> str:
        return f"Z{i+1}"

    def zoneIndices(self, values: np.ndarray) -> np.ndarray:
        """Zone index for each of ``values`` — the whole classification rule.

        ``validate`` guarantees the bounds are ordered and non-overlapping, so
        the first zone whose upper bound reaches a value is the zone that owns
        it. Values that fall short of that zone's lower bound are in a gap
        (or below Z1) and belong to the nearest zone *below*, not the top one:
        falling through to the last zone booked easy samples as maximal effort,
        a 3 W gap between Z1 and Z2 filing recovery-pace riding as Z7. New gaps
        are rejected on write, but snapshots are backfilled from zone lists
        saved before that rule existed.

        This is the only place the rule is expressed; ``getZone`` is the scalar
        entry point onto it.
        """
        last = len(self.zones) - 1
        lowers = np.array([lower for lower, _ in self.zones])
        uppers = np.array([upper for _, upper in self.zones])

        i = np.minimum(np.searchsorted(uppers, values, side="left"), last)
        i = np.where(values < lowers[i], i - 1, i)
        return np.clip(i, 0, last)

    def getZone(self, v: int) -> int:
        return int(self.zoneIndices(np.array([v]))[0])


    def validate(self) -> None:
        for i, (lower, upper) in enumerate(self.zones):
            if upper <= lower:
                raise ValueError(
                    f"{self.zoneName(i)} is invalid: upper bound ({upper}) must be greater than lower bound ({lower})"
                )

            if i < len(self.zones) - 1:
                next_lower = self.zones[i + 1][0]
                if upper > next_lower:
                    raise ValueError(
                        f"{self.zoneName(i)} is invalid: upper bound ({upper}) must be lower than "
                        f"{self.zoneName(i+1)} lower bound ({next_lower})"
                    )

