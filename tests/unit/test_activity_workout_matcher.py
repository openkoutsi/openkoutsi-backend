"""Unit tests for the activity→planned workout matching service."""
import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone

from backend.app.services.activity_workout_matcher import _matches
from backend.app.models.team_orm import Activity, PlannedWorkout


def _activity(**kwargs) -> Activity:
    a = MagicMock(spec=Activity)
    a.sport_type = kwargs.get("sport_type", "Ride")
    a.tss = kwargs.get("tss", None)
    a.duration_s = kwargs.get("duration_s", None)
    a.start_time = kwargs.get("start_time", datetime(2025, 6, 2, 10, 0, tzinfo=timezone.utc))
    return a


def _workout(**kwargs) -> PlannedWorkout:
    w = MagicMock(spec=PlannedWorkout)
    w.workout_type = kwargs.get("workout_type", "threshold")
    w.target_tss = kwargs.get("target_tss", None)
    w.duration_min = kwargs.get("duration_min", None)
    w.completed_activity_id = kwargs.get("completed_activity_id", None)
    return w


class TestMatchesFunction:
    def test_sport_mismatch_returns_false(self):
        act = _activity(sport_type="Run")
        wo = _workout(workout_type="swim")
        assert _matches(act, wo) is False

    def test_sport_match_no_tss_no_duration_returns_true(self):
        act = _activity(sport_type="Ride")
        wo = _workout(workout_type="threshold", target_tss=None, duration_min=None)
        assert _matches(act, wo) is True

    def test_tss_above_threshold_passes(self):
        act = _activity(sport_type="Ride", tss=65.0)
        wo = _workout(workout_type="threshold", target_tss=100, duration_min=None)
        assert _matches(act, wo) is True  # 65 >= 60% of 100

    def test_tss_exactly_at_threshold_passes(self):
        act = _activity(sport_type="Ride", tss=60.0)
        wo = _workout(workout_type="threshold", target_tss=100, duration_min=None)
        assert _matches(act, wo) is True

    def test_tss_below_threshold_fails(self):
        act = _activity(sport_type="Ride", tss=59.0)
        wo = _workout(workout_type="threshold", target_tss=100, duration_min=None)
        assert _matches(act, wo) is False

    def test_tss_none_treated_as_zero_fails_when_target_set(self):
        act = _activity(sport_type="Ride", tss=None)
        wo = _workout(workout_type="threshold", target_tss=100, duration_min=None)
        assert _matches(act, wo) is False

    def test_tss_none_passes_when_no_target(self):
        act = _activity(sport_type="Ride", tss=None)
        wo = _workout(workout_type="easy", target_tss=None, duration_min=None)
        assert _matches(act, wo) is True

    def test_duration_above_threshold_passes(self):
        act = _activity(sport_type="Ride", duration_s=3600)  # 60 min
        wo = _workout(workout_type="easy", duration_min=60, target_tss=None)
        assert _matches(act, wo) is True  # 3600 >= 60% of 3600

    def test_duration_below_threshold_fails(self):
        act = _activity(sport_type="Ride", duration_s=2159)  # 35.98 min < 60% of 60 min
        wo = _workout(workout_type="easy", duration_min=60, target_tss=None)
        assert _matches(act, wo) is False

    def test_all_criteria_pass(self):
        act = _activity(sport_type="Ride", tss=80.0, duration_s=4500)
        wo = _workout(workout_type="threshold", target_tss=100, duration_min=60)
        assert _matches(act, wo) is True

    def test_virtual_ride_matches_threshold_workout(self):
        act = _activity(sport_type="VirtualRide", tss=90.0, duration_s=5400)
        wo = _workout(workout_type="threshold", target_tss=100, duration_min=60)
        assert _matches(act, wo) is True

    def test_walk_does_not_match_endurance_workout(self):
        act = _activity(sport_type="Walk", tss=50.0, duration_s=7200)
        wo = _workout(workout_type="endurance", target_tss=60, duration_min=90)
        assert _matches(act, wo) is False
