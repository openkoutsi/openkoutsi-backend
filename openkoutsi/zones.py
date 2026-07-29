from dataclasses import dataclass
from typing import Iterable, List, Sequence

# The canonical zone models. Zone lists used to be arbitrary-length, which made
# anything built on top of them (see ``intensity_distribution``) guess at what a
# given zone meant. They are now fixed: seven power zones (Coggan) and five HR
# zones, both running Z1 (recovery) → Zn (maximal). The API rejects other
# lengths; snapshots frozen before that still carry whatever count they were
# computed with, so readers must degrade rather than assume.
POWER_ZONE_COUNT = 7  # Z1 Recovery … Z7 Neuromuscular
HR_ZONE_COUNT = 5  # Z1 Recovery … Z5 VO2max


def time_in_zones(samples: Iterable[float], zone_defs: Sequence[dict]) -> dict[str, int]:
    """Accumulate time spent in each zone from a per-second sample stream.

    ``samples`` is a 1 Hz stream (one value per second), so each sample counts
    as one second. ``zone_defs`` is the athlete's zone list — ``[{"low", "high",
    "name"}, ...]``. Returns ``{zone_name: seconds}``. Values below Z1 / above
    the last zone are clamped into the nearest zone by ``Zones.getZone``.
    """
    zones = Zones(*[(z["low"], z["high"]) for z in zone_defs])
    out: dict[str, int] = {}
    for v in samples:
        i = zones.getZone(int(v))
        name = zone_defs[i].get("name", f"Z{i + 1}")
        out[name] = out.get(name, 0) + 1
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
    
    def getZone(self, v: int) -> int:
        for i, (lower, upper) in enumerate(self.zones):
            if v >= lower and v <= upper:
                return i
        # Below Z1 → clamp to Z1; above last zone → clamp to last zone.
        if v < self.zones[0][0]:
            return 0
        return len(self.zones) - 1


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

