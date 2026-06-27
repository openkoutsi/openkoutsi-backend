from dataclasses import dataclass
from datetime import datetime
import json

from .zones import Zones


@dataclass
class Profile:
    start_time: datetime
    duration: int  # seconds
    distance: int  # metres
    elevationGain: int  # metres
    avgHeartRate: float  # BPM
    avgSpeed: float  # km/h
    avgPower: float  # W
    peakPower: float  # W
    peakHR: float  # BPM
    peakCadence: float  # RPM
    avgCadence: float  # RPM
    heartRate: list[float]  # BPM at each second
    speed: list[float]  # km/h at each second
    power: list[float]  # W at each second
    cadence: list[float]  # RPM at each second
    altitude: list[float]  # metres at each second

    sport_type: str | None  # raw sport string from FIT file, e.g. "running"

    def __init__(
        self,
        start_time: datetime,
        duration: int,
        distance: int,
        elevationGain: int,
        heartRate: list[float],
        speed: list[float],
        power: list[float],
        cadence: list[float],
        altitude: list[float] | None = None,
        sport_type: str | None = None,
    ):
        self.start_time = start_time
        self.duration = duration
        self.distance = distance
        self.elevationGain = elevationGain
        self.heartRate = heartRate
        self.speed = speed
        self.power = power
        self.cadence = cadence
        self.altitude = altitude or []
        self.sport_type = sport_type

        self.avgHeartRate = (sum(heartRate) / len(heartRate)) if heartRate else 0.0
        self.avgSpeed = (sum(speed) / len(speed)) if speed else 0.0
        self.avgPower = (sum(power) / len(power)) if power else 0.0
        self.peakPower = max(power) if power else 0
        self.peakHR = max(heartRate) if heartRate else 0
        self.peakCadence = max(cadence) if cadence else 0
        self.avgCadence = int(round(sum(cadence) / len(cadence))) if cadence else 0

    @classmethod
    def from_json(cls, value: str | dict) -> "Profile":
        data = json.loads(value) if isinstance(value, str) else value
        return cls(
            start_time=datetime.fromisoformat(data["start_time"]),
            duration=data["duration"],
            distance=data["distance"],
            elevationGain=data["elevationGain"],
            heartRate=data["heartRate"],
            speed=data["speed"],
            power=data["power"],
            cadence=data.get("cadence", []),
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "start_time": self.start_time.isoformat(),
                "duration": self.duration,
                "distance": self.distance,
                "elevationGain": self.elevationGain,
                "avgHeartRate": self.avgHeartRate,
                "avgSpeed": self.avgSpeed,
                "avgPower": self.avgPower,
                "peakPower": self.peakPower,
                "peakHR": self.peakHR,
                "peakCadence": self.peakCadence,
                "avgCadence": self.avgCadence,
                "heartRate": self.heartRate,
                "speed": self.speed,
                "power": self.power,
                "cadence": self.cadence,
            }
        )


def zoneBreakdown(
    workout: Profile, hrZones: Zones, powerZones: Zones
) -> tuple[dict, dict]:
    """Analyze a workout profile and summarize time spent in heart-rate and power zones.
    This function iterates through the given workout profile and accumulates the
    amount of time spent in each HR zone and each power zone.
    Args:
        workout: The workout profile to analyze.
        hrZones: The configured heart-rate zones used for classification.
        powerZones: The configured power zones used for classification.
        Returns:
            A tuple of two dictionaries:
                - The first dictionary maps each heart-rate zone to the time spent in it.
                - The second dictionary maps each power zone to the time spent in it.
    """

    sample_count = min(
        workout.duration, len(workout.heartRate), len(workout.speed), len(workout.power)
    )
    timeInHrZones = {hrZones.zoneName(i): 0 for i in range(len(hrZones.zones))}
    timeInPowerZones = {powerZones.zoneName(i): 0 for i in range(len(powerZones.zones))}

    for i in range(sample_count):
        hr_zone = hrZones.getZone(int(workout.heartRate[i]))
        power_zone = powerZones.getZone(int(workout.power[i]))
        hr_zone_name = hrZones.zoneName(hr_zone)
        power_zone_name = powerZones.zoneName(power_zone)

        timeInHrZones[hr_zone_name] = timeInHrZones.get(hr_zone_name, 0) + 1
        timeInPowerZones[power_zone_name] = timeInPowerZones.get(power_zone_name, 0) + 1

    return (timeInHrZones, timeInPowerZones)
