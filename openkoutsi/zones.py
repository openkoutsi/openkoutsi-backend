from dataclasses import dataclass
from typing import List

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

