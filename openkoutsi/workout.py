from dataclasses import dataclass
from datetime import datetime
import json

import numpy as np


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

        hr = np.asarray(heartRate, dtype=float)
        spd = np.asarray(speed, dtype=float)
        pwr = np.asarray(power, dtype=float)
        cad = np.asarray(cadence, dtype=float)

        self.avgHeartRate = float(hr.mean()) if hr.size else 0.0
        self.avgSpeed = float(spd.mean()) if spd.size else 0.0
        self.avgPower = float(pwr.mean()) if pwr.size else 0.0
        self.peakPower = float(pwr.max()) if pwr.size else 0
        self.peakHR = float(hr.max()) if hr.size else 0
        self.peakCadence = float(cad.max()) if cad.size else 0
        self.avgCadence = int(round(float(cad.mean()))) if cad.size else 0

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
