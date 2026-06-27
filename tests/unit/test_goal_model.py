import json
import pytest
from datetime import date
from openkoutsi.goal import Goal


class TestGoal:
    def test_from_json_dict(self):
        g = Goal.from_json({"goalType": "race", "description": "Gran Fondo", "date": "2025-09-01", "target": "finish"})
        assert g.goalType == "race"
        assert g.description == "Gran Fondo"
        assert g.date == date(2025, 9, 1)
        assert g.target == "finish"

    def test_from_json_string(self):
        g = Goal.from_json('{"goalType": "fitness", "description": "Lose weight", "date": "2025-12-31", "target": "75kg"}')
        assert g.goalType == "fitness"
        assert g.date == date(2025, 12, 31)

    def test_from_string_alias(self):
        raw = '{"goalType": "race", "description": "Century ride", "date": "2025-08-01", "target": "sub-5h"}'
        g1 = Goal.from_json(raw)
        g2 = Goal.from_string(raw)
        assert g1.goalType == g2.goalType
        assert g1.date == g2.date

    def test_to_json(self):
        g = Goal(goalType="performance", description="Test", date=date(2025, 6, 15), target="300W FTP")
        data = json.loads(g.to_json())
        assert data["goalType"] == "performance"
        assert data["description"] == "Test"
        assert data["date"] == "2025-06-15"
        assert data["target"] == "300W FTP"

    def test_roundtrip(self):
        original = Goal(goalType="race", description="Tour de Suisse", date=date(2025, 7, 4), target="top-50")
        restored = Goal.from_json(original.to_json())
        assert restored.goalType == original.goalType
        assert restored.description == original.description
        assert restored.date == original.date
        assert restored.target == original.target
