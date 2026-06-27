import json
import pytest
from datetime import date
from openkoutsi.athlete import Athlete, FtpTest, Availability
from openkoutsi.goal import Goal


def _availability_dict():
    return {
        "sessionsPerWeek": 5,
        "maxSessionHours": 2.0,
        "longRideDay": "Saturday",
        "details": "Morning sessions preferred",
    }


def _goal_dict():
    return {"goalType": "race", "description": "Gran Fondo", "date": "2025-09-01", "target": "finish"}


def _athlete_dict(hr_zones=None, power_zones=None, ftp_tests=None, goals=None):
    return {
        "maxHR": 185,
        "weight": 72.5,
        "currentFTP": 280,
        "hrZones": hr_zones,
        "powerZones": power_zones,
        "ftp_tests": ftp_tests or [],
        "availability": _availability_dict(),
        "goals": goals or [],
    }


class TestFtpTest:
    def test_from_json_dict(self):
        ft = FtpTest.from_json({"date": "2025-01-15", "ftp": 270})
        assert ft.date == date(2025, 1, 15)
        assert ft.ftp == 270

    def test_from_json_string(self):
        ft = FtpTest.from_json('{"date": "2025-03-01", "ftp": 285}')
        assert ft.date == date(2025, 3, 1)
        assert ft.ftp == 285

    def test_to_json_roundtrip(self):
        ft = FtpTest(date=date(2025, 6, 10), ftp=295)
        data = json.loads(ft.to_json())
        assert data["date"] == "2025-06-10"
        assert data["ftp"] == 295

    def test_from_to_json_roundtrip(self):
        original = FtpTest(date=date(2024, 12, 1), ftp=260)
        restored = FtpTest.from_json(original.to_json())
        assert restored.date == original.date
        assert restored.ftp == original.ftp


class TestAvailability:
    def test_from_json_dict(self):
        av = Availability.from_json(_availability_dict())
        assert av.sessionsPerWeek == 5
        assert av.maxSessionHours == 2.0
        assert av.longRideDay == "Saturday"
        assert av.details == "Morning sessions preferred"

    def test_from_json_string(self):
        av = Availability.from_json(json.dumps(_availability_dict()))
        assert av.sessionsPerWeek == 5

    def test_to_json_roundtrip(self):
        av = Availability.from_json(_availability_dict())
        data = json.loads(av.to_json())
        assert data["sessionsPerWeek"] == 5
        assert data["longRideDay"] == "Saturday"

    def test_from_to_roundtrip(self):
        original = Availability.from_json(_availability_dict())
        restored = Availability.from_json(original.to_json())
        assert restored.sessionsPerWeek == original.sessionsPerWeek
        assert restored.maxSessionHours == original.maxSessionHours


class TestAthlete:
    def test_from_json_minimal(self):
        data = _athlete_dict()
        a = Athlete.from_json(data)
        assert a.maxHR == 185
        assert a.weight == 72.5
        assert a.currentFTP == 280
        assert a.hrZones is None
        assert a.powerZones is None
        assert a.ftp_tests == []
        assert a.goals == []

    def test_from_json_with_zones(self):
        data = _athlete_dict(
            hr_zones=[[0, 120], [121, 148], [149, 165]],
            power_zones=[[0, 150], [151, 210], [211, 250]],
        )
        a = Athlete.from_json(data)
        assert a.hrZones is not None
        assert a.powerZones is not None
        assert a.hrZones.zones[0] == (0, 120)
        assert a.powerZones.zones[2] == (211, 250)

    def test_from_json_with_ftp_tests(self):
        data = _athlete_dict(ftp_tests=[{"date": "2025-01-01", "ftp": 260}])
        a = Athlete.from_json(data)
        assert len(a.ftp_tests) == 1
        assert a.ftp_tests[0].ftp == 260

    def test_from_json_with_goals(self):
        data = _athlete_dict(goals=[_goal_dict()])
        a = Athlete.from_json(data)
        assert len(a.goals) == 1
        assert a.goals[0].goalType == "race"

    def test_to_json_roundtrip(self):
        data = _athlete_dict(
            hr_zones=[[0, 120], [121, 148]],
            ftp_tests=[{"date": "2025-03-01", "ftp": 280}],
            goals=[_goal_dict()],
        )
        a = Athlete.from_json(data)
        serialized = json.loads(a.to_json())
        assert serialized["maxHR"] == 185
        assert serialized["currentFTP"] == 280
        assert serialized["hrZones"] == [[0, 120], [121, 148]]
        assert serialized["powerZones"] is None
        assert len(serialized["ftp_tests"]) == 1
        assert len(serialized["goals"]) == 1

    def test_from_json_string(self):
        data = _athlete_dict()
        a = Athlete.from_json(json.dumps(data))
        assert a.maxHR == 185
