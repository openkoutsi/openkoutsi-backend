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
    async def test_returns_none_without_plan(self, session, seeded_athlete):
        act = await _persist_activity(session, seeded_athlete.id)
        assert await resolve_planned_workout_for_activity(
            session, seeded_athlete.id, act
        ) is None

    async def test_returns_scheduled_workout_for_date_no_threshold(
        self, session, seeded_athlete
    ):
        # Below the auto-match load threshold, but still surfaced for the day.
        w = PlannedWorkout(
            week_number=1, day_of_week=1, workout_type="threshold",
            target_load=100, duration_min=60,
        )
        await _seed_plan(session, seeded_athlete.id, [w])
        act = await _persist_activity(session, seeded_athlete.id)
        act.load = 10.0  # nowhere near 60% of target
        resolved = await resolve_planned_workout_for_activity(
            session, seeded_athlete.id, act
        )
        assert resolved is not None
        assert resolved.id == w.id

    async def test_prefers_sport_matching_candidate(self, session, seeded_athlete):
        run = PlannedWorkout(week_number=1, day_of_week=1, workout_type="run")
        ride = PlannedWorkout(week_number=1, day_of_week=1, workout_type="threshold")
        await _seed_plan(session, seeded_athlete.id, [run, ride])
        act = await _persist_activity(session, seeded_athlete.id, sport_type="Ride")
        resolved = await resolve_planned_workout_for_activity(
            session, seeded_athlete.id, act
        )
        assert resolved is not None
        assert resolved.id == ride.id

    async def test_prefers_linked_workout_over_date(self, session, seeded_athlete):
        # A workout on Wednesday (day 3); the activity is on Monday (day 1) but is
        # explicitly linked to the Wednesday workout → the link wins.
        mon = PlannedWorkout(week_number=1, day_of_week=1, workout_type="threshold")
        wed = PlannedWorkout(week_number=1, day_of_week=3, workout_type="endurance")
        await _seed_plan(session, seeded_athlete.id, [mon, wed])
        act = await _persist_activity(session, seeded_athlete.id)
        session.add(PlannedWorkoutActivity(
            planned_workout_id=wed.id, activity_id=act.id
        ))
        await session.commit()
        resolved = await resolve_planned_workout_for_activity(
            session, seeded_athlete.id, act
        )
        assert resolved is not None
        assert resolved.id == wed.id

    async def test_returns_none_for_date_outside_plan(self, session, seeded_athlete):
        w = PlannedWorkout(week_number=1, day_of_week=1, workout_type="threshold")
        await _seed_plan(session, seeded_athlete.id, [w])
        # Activity a week before the plan starts.
        act = await _persist_activity(
            session, seeded_athlete.id,
            start=datetime(2025, 5, 26, 10, tzinfo=timezone.utc),
        )
        assert await resolve_planned_workout_for_activity(
            session, seeded_athlete.id, act
        ) is None
