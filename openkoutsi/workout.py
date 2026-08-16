from dataclasses import dataclass
from datetime import datetime
import json

import numpy as np

from . import streams as stream_utils


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
    # Streams on a common 1 Hz clock: index i is second i from the first record,
    # in every channel, with None where that channel had no sample. See
    # ``openkoutsi.streams`` for the contract. An empty list means the activity
    # has no such channel at all, which is not the same as a channel of gaps.
    heartRate: list[float | None]  # BPM at each second
    speed: list[float | None]  # km/h at each second
    power: list[float | None]  # W at each second
    cadence: list[float | None]  # RPM at each second
    altitude: list[float | None]  # metres at each second

    sport_type: str | None  # raw sport string from FIT file, e.g. "running"
    # The name the file gave the activity, when it carries one. GPX and TCX both
    # do (Strava writes the ride's title into the track), and using it beats
    # calling nine hundred imported rides "Uploaded Activity". FIT files rarely
    # carry one, so this stays None on that path.
    name: str | None

    def __init__(
        self,
        start_time: datetime,
        duration: int,
        distance: int,
        elevationGain: int,
        heartRate: list[float | None],
        speed: list[float | None],
        power: list[float | None],
        cadence: list[float | None],
        altitude: list[float | None] | None = None,
        sport_type: str | None = None,
        name: str | None = None,
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
        self.name = name

        # Averaged over the samples that exist, not over the grid: a strap that
        # dropped for ten minutes should not pull average HR toward zero. This
        # is what these figures meant before the streams carried gaps, when a
        # dropout simply shortened the list.
        hr = stream_utils.present(heartRate)
        spd = stream_utils.present(speed)
        pwr = stream_utils.present(power)
        cad = stream_utils.present(cadence)

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
                # Gaps as JSON null: ``json.dumps`` would happily emit a bare
                # ``NaN`` token, which is not valid JSON and which Postgres
                # rejects even though SQLite accepts it.
                "heartRate": stream_utils.to_json_stream(self.heartRate),
                "speed": stream_utils.to_json_stream(self.speed),
                "power": stream_utils.to_json_stream(self.power),
                "cadence": stream_utils.to_json_stream(self.cadence),
            }
        )
