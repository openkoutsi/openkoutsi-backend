from dataclasses import dataclass
from datetime import date as d
from typing import List
import json


from .zones import Zones
from .goal import Goal


@dataclass
class FtpTest:
    date: d
    ftp: int  # W

    @classmethod
    def from_json(cls, value: str | dict) -> "FtpTest":
        data = json.loads(value) if isinstance(value, str) else value
        return cls(
            date=d.fromisoformat(data["date"]),
            ftp=data["ftp"],
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "date": self.date.isoformat(),
                "ftp": self.ftp,
            }
        )


@dataclass
class Availability:
    sessionsPerWeek: int
    maxSessionHours: float
    longRideDay: str
    details: str

    @classmethod
    def from_json(cls, value: str | dict) -> "Availability":
        data = json.loads(value) if isinstance(value, str) else value
        return cls(
            sessionsPerWeek=data["sessionsPerWeek"],
            maxSessionHours=data["maxSessionHours"],
            longRideDay=data["longRideDay"],
            details=data["details"],
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "sessionsPerWeek": self.sessionsPerWeek,
                "maxSessionHours": self.maxSessionHours,
                "longRideDay": self.longRideDay,
                "details": self.details,
            }
        )


@dataclass
class Athlete:
    maxHR: int  # BPM
    weight: float  # kg
    currentFTP: int  # W
    hrZones: Zones | None
    powerZones: Zones | None
    ftp_tests: List[FtpTest]
    availability: Availability
    goals: List[Goal]

    @classmethod
    def from_json(cls, value: str | dict) -> "Athlete":
        data = json.loads(value) if isinstance(value, str) else value

        hr_zones_data = data.get("hrZones")
        power_zones_data = data.get("powerZones")

        hr_zones = (
            Zones(*[tuple(zone) for zone in hr_zones_data])
            if hr_zones_data is not None
            else None
        )
        power_zones = (
            Zones(*[tuple(zone) for zone in power_zones_data])
            if power_zones_data is not None
            else None
        )

        return cls(
            maxHR=data["maxHR"],
            weight=data["weight"],
            currentFTP=data["currentFTP"],
            hrZones=hr_zones,
            powerZones=power_zones,
            ftp_tests=[FtpTest.from_json(item) for item in data["ftp_tests"]],
            availability=Availability.from_json(data["availability"]),
            goals=[
                Goal.from_json(item if isinstance(item, str) else json.dumps(item))
                for item in data["goals"]
            ],
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "maxHR": self.maxHR,
                "weight": self.weight,
                "currentFTP": self.currentFTP,
                "hrZones": self.hrZones.zones if self.hrZones is not None else None,
                "powerZones": (
                    self.powerZones.zones if self.powerZones is not None else None
                ),
                "ftp_tests": [json.loads(item.to_json()) for item in self.ftp_tests],
                "availability": json.loads(self.availability.to_json()),
                "goals": [json.loads(item.to_json()) for item in self.goals],
            }
        )
