"""Unit tests for the activity→planned workout matching service."""
import pytest
from unittest.mock import MagicMock
from datetime import date, datetime, timedelta, timezone

from backend.app.services.activity_workout_matcher import (
    _matches,
    resolve_planned_workout_for_activity,
)
from backend.app.models.user_orm import (
    Activity,
    PlannedWorkout,
    PlannedWorkoutActivity,
    TrainingPlan,
)


def _activity(**kwargs) -> Activity:
    a = MagicMock(spec=Activity)
    a.sport_type = kwargs.get("sport_type", "Ride")
    a.load = kwargs.get("load", None)
    a.duration_s = kwargs.get("duration_s", None)
    a.start_time = kwargs.get("start_time", datetime(2025, 6, 2, 10, 0, tzinfo=timezone.utc))
    return a


def _workout(**kwargs) -> PlannedWorkout:
    w = MagicMock(spec=PlannedWorkout)
    w.workout_type = kwargs.get("workout_type", "threshold")
    w.target_load = kwargs.get("target_load", None)
    w.duration_min = kwargs.get("duration_min", None)
    return w


class TestMatchesFunction:
    def test_sport_mismatch_returns_false(self):
        act = _activity(sport_type="Run")
        wo = _workout(workout_type="swim")
        assert _matches(act, wo) is False

    def test_sport_match_no_tss_no_duration_returns_true(self):
        act = _activity(sport_type="Ride")
        wo = _workout(workout_type="threshold", target_load=None, duration_min=None)
        assert _matches(act, wo) is True

    def test_tss_above_threshold_passes(self):
        act = _activity(sport_type="Ride", load=65.0)
        wo = _workout(workout_type="threshold", target_load=100, duration_min=None)
        assert _matches(act, wo) is True  # 65 >= 60% of 100

    def test_tss_exactly_at_threshold_passes(self):
        act = _activity(sport_type="Ride", load=60.0)
        wo = _workout(workout_type="threshold", target_load=100, duration_min=None)
        assert _matches(act, wo) is True

    def test_tss_below_threshold_fails(self):
        act = _activity(sport_type="Ride", load=59.0)
        wo = _workout(workout_type="threshold", target_load=100, duration_min=None)
        assert _matches(act, wo) is False

    def test_tss_none_treated_as_zero_fails_when_target_set(self):
        act = _activity(sport_type="Ride", load=None)
        wo = _workout(workout_type="threshold", target_load=100, duration_min=None)
        assert _matches(act, wo) is False

    def test_tss_none_passes_when_no_target(self):
        act = _activity(sport_type="Ride", load=None)
        wo = _workout(workout_type="easy", target_load=None, duration_min=None)
        assert _matches(act, wo) is True

    def test_duration_above_threshold_passes(self):
        act = _activity(sport_type="Ride", duration_s=3600)  # 60 min
        wo = _workout(workout_type="easy", duration_min=60, target_load=None)
        assert _matches(act, wo) is True  # 3600 >= 60% of 3600

    def test_duration_below_threshold_fails(self):
        act = _activity(sport_type="Ride", duration_s=2159)  # 35.98 min < 60% of 60 min
        wo = _workout(workout_type="easy", duration_min=60, target_load=None)
        assert _matches(act, wo) is False

    def test_all_criteria_pass(self):
        act = _activity(sport_type="Ride", load=80.0, duration_s=4500)
        wo = _workout(workout_type="threshold", target_load=100, duration_min=60)
        assert _matches(act, wo) is True

    def test_virtual_ride_matches_threshold_workout(self):
        act = _activity(sport_type="VirtualRide", load=90.0, duration_s=5400)
        wo = _workout(workout_type="threshold", target_load=100, duration_min=60)
        assert _matches(act, wo) is True

    def test_walk_does_not_match_endurance_workout(self):
        act = _activity(sport_type="Walk", load=50.0, duration_s=7200)
        wo = _workout(workout_type="endurance", target_load=60, duration_min=90)
        assert _matches(act, wo) is False


# ── resolve_planned_workout_for_activity ──────────────────────────────────────

_START = date(2025, 6, 2)  # A Monday → day_of_week 1, week_number 1


async def _seed_plan(session, athlete_id, workouts, *, start=_START):
    plan = TrainingPlan(
        athlete_id=athlete_id, name="P", start_date=start,
        end_date=start + timedelta(weeks=2), status="active",
    )
    session.add(plan)
    await session.flush()
    for w in workouts:
        w.plan_id = plan.id
        session.add(w)
    await session.commit()
    return plan


async def _persist_activity(session, athlete_id, *, sport_type="Ride", start=None):
    act = Activity(
        athlete_id=athlete_id, sport_type=sport_type,
        start_time=start or datetime(2025, 6, 2, 10, tzinfo=timezone.utc),
    )
    session.add(act)
    await session.commit()
    return act


class TestResolvePlannedWorkoutForActivity:
    async def test_returns_none_when_not_linked(self, session, seeded_athlete):
        # A planned workout exists for the activity's day, but the activity is
        # not linked to it → nothing is surfaced (no date-based guessing).
        w = PlannedWorkout(week_number=1, day_of_week=1, workout_type="threshold")
        await _seed_plan(session, seeded_athlete.id, [w])
        act = await _persist_activity(session, seeded_athlete.id)
        assert await resolve_planned_workout_for_activity(session, act) is None

    async def test_returns_linked_workout(self, session, seeded_athlete):
        w = PlannedWorkout(week_number=1, day_of_week=1, workout_type="threshold")
        await _seed_plan(session, seeded_athlete.id, [w])
        act = await _persist_activity(session, seeded_athlete.id)
        session.add(PlannedWorkoutActivity(
            planned_workout_id=w.id, activity_id=act.id
        ))
        await session.commit()
        resolved = await resolve_planned_workout_for_activity(session, act)
        assert resolved is not None
        assert resolved.id == w.id

    async def test_ignores_link_belonging_to_another_activity(
        self, session, seeded_athlete
    ):
        # The reported bug: the day's key session is completed by one ride; a
        # later unlinked commute spin must NOT be evaluated against it.
        w = PlannedWorkout(week_number=1, day_of_week=1, workout_type="threshold")
        await _seed_plan(session, seeded_athlete.id, [w])
        key_ride = await _persist_activity(session, seeded_athlete.id)
        session.add(PlannedWorkoutActivity(
            planned_workout_id=w.id, activity_id=key_ride.id
        ))
        await session.commit()
        commute = await _persist_activity(session, seeded_athlete.id)
        assert await resolve_planned_workout_for_activity(session, commute) is None
