"""Unit tests for the LLM training status prompt builder."""
from datetime import date, datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

from backend.app.models.user_orm import Athlete, PlannedWorkout, TrainingPlan
from backend.app.services.llm_training_status_analyzer import (
    _build_status_prompt,
    _build_system_prompt,
)


def _athlete(app_settings: dict | None = None) -> Athlete:
    a = MagicMock(spec=Athlete)
    a.ftp = 250
    a.max_hr = 190
    a.app_settings = app_settings if app_settings is not None else {}
    return a


def _plan(start: date, *, name: str = "Base build", end: date | None = None) -> TrainingPlan:
    p = MagicMock(spec=TrainingPlan)
    p.id = "plan-1"
    p.name = name
    p.start_date = start
    p.end_date = end
    return p


def _workout(day_of_week: int, **kwargs) -> PlannedWorkout:
    w = MagicMock(spec=PlannedWorkout)
    w.day_of_week = day_of_week
    w.workout_type = kwargs.get("workout_type", "endurance")
    w.target_load = kwargs.get("target_load", None)
    linked = kwargs.get("linked_activities", [])
    w.linked_activities = linked
    w.is_completed = bool(linked)
    w.skip_reason = kwargs.get("skip_reason", None)
    return w


def _linked_activity(load=50.0, duration_s=3600):
    a = MagicMock()
    a.load = load
    a.duration_s = duration_s
    return a


class TestThisWeekWeekdayLabels:
    def test_workouts_labelled_with_weekday_and_date(self):
        # Plan starts Monday 2025-06-02; "now" is Wednesday 2025-06-04.
        plan_start = date(2025, 6, 2)
        now = datetime(2025, 6, 4, 8, 0, tzinfo=ZoneInfo("Europe/Helsinki"))
        workouts = [
            _workout(1, workout_type="threshold", target_load=80),  # Monday
            _workout(3, workout_type="intervals", target_load=90),  # Wednesday = today
            _workout(7, workout_type="long", target_load=120),       # Sunday
        ]
        prompt = _build_status_prompt(
            athlete=_athlete(),
            recent_activities=[],
            current_metric=None,
            active_plans=[(_plan(plan_start), workouts)],
            active_goals=[],
            now=now,
        )
        # Explicit weekday names and ISO dates, not "Day 1/3/7".
        assert "Monday 2025-06-02" in prompt
        assert "Wednesday 2025-06-04 (today)" in prompt
        assert "Sunday 2025-06-08" in prompt
        assert "Day 1:" not in prompt

    def test_handles_plan_starting_midweek(self):
        # Plan starts Thursday 2025-06-05; week 1 block is Thu..Wed.
        plan_start = date(2025, 6, 5)
        now = datetime(2025, 6, 6, 8, 0, tzinfo=ZoneInfo("UTC"))  # Friday, week 1
        workouts = [
            _workout(5, workout_type="tempo"),  # Friday 2025-06-06 (today)
            _workout(2, workout_type="rest"),   # Tuesday 2025-06-10 (next week-day in block)
        ]
        prompt = _build_status_prompt(
            athlete=_athlete(),
            recent_activities=[],
            current_metric=None,
            active_plans=[(_plan(plan_start), workouts)],
            active_goals=[],
            now=now,
        )
        assert "Friday 2025-06-06 (today)" in prompt
        # Tuesday falls on the second-to-last day of the Thu..Wed block.
        assert "Tuesday 2025-06-10" in prompt


class TestSkipReason:
    def test_skip_reason_included_for_incomplete_workout(self):
        plan_start = date(2025, 6, 2)
        now = datetime(2025, 6, 4, 8, 0, tzinfo=ZoneInfo("Europe/Helsinki"))
        workouts = [
            _workout(1, workout_type="threshold", skip_reason="Feeling sick"),
        ]
        prompt = _build_status_prompt(
            athlete=_athlete(),
            recent_activities=[],
            current_metric=None,
            active_plans=[(_plan(plan_start), workouts)],
            active_goals=[],
            now=now,
        )
        assert "not completed (skipped — reason: Feeling sick)" in prompt

    def test_skip_reason_omitted_when_completed(self):
        plan_start = date(2025, 6, 2)
        now = datetime(2025, 6, 4, 8, 0, tzinfo=ZoneInfo("Europe/Helsinki"))
        workouts = [
            _workout(
                1,
                workout_type="threshold",
                linked_activities=[_linked_activity()],
                skip_reason="stale reason",
            ),
        ]
        prompt = _build_status_prompt(
            athlete=_athlete(),
            recent_activities=[],
            current_metric=None,
            active_plans=[(_plan(plan_start), workouts)],
            active_goals=[],
            now=now,
        )
        assert "completed" in prompt
        assert "skipped" not in prompt
        assert "stale reason" not in prompt

    def test_no_skip_annotation_without_reason(self):
        plan_start = date(2025, 6, 2)
        now = datetime(2025, 6, 4, 8, 0, tzinfo=ZoneInfo("Europe/Helsinki"))
        workouts = [_workout(1, workout_type="threshold")]
        prompt = _build_status_prompt(
            athlete=_athlete(),
            recent_activities=[],
            current_metric=None,
            active_plans=[(_plan(plan_start), workouts)],
            active_goals=[],
            now=now,
        )
        assert "not completed" in prompt
        assert "skipped" not in prompt


class TestRestDays:
    def test_rest_day_not_rendered_as_incomplete(self):
        # A rest day earlier in the week must not read as a missed session.
        plan_start = date(2025, 6, 2)  # Monday
        now = datetime(2025, 6, 4, 8, 0, tzinfo=ZoneInfo("Europe/Helsinki"))  # Wed
        workouts = [_workout(1, workout_type="rest")]  # Monday, in the past
        prompt = _build_status_prompt(
            athlete=_athlete(),
            recent_activities=[],
            current_metric=None,
            active_plans=[(_plan(plan_start), workouts)],
            active_goals=[],
            now=now,
        )
        assert "rest day — nothing to complete, no action required" in prompt
        assert "not completed" not in prompt

    def test_rest_day_ignores_skip_reason(self):
        # Rest days carry no skip semantics even if a stale reason is attached.
        plan_start = date(2025, 6, 2)
        now = datetime(2025, 6, 4, 8, 0, tzinfo=ZoneInfo("Europe/Helsinki"))
        workouts = [_workout(1, workout_type="rest", skip_reason="stale reason")]
        prompt = _build_status_prompt(
            athlete=_athlete(),
            recent_activities=[],
            current_metric=None,
            active_plans=[(_plan(plan_start), workouts)],
            active_goals=[],
            now=now,
        )
        assert "rest day — nothing to complete, no action required" in prompt
        assert "skipped" not in prompt
        assert "stale reason" not in prompt

    def test_rest_day_case_insensitive(self):
        plan_start = date(2025, 6, 2)
        now = datetime(2025, 6, 4, 8, 0, tzinfo=ZoneInfo("Europe/Helsinki"))
        workouts = [_workout(1, workout_type="Rest")]
        prompt = _build_status_prompt(
            athlete=_athlete(),
            recent_activities=[],
            current_metric=None,
            active_plans=[(_plan(plan_start), workouts)],
            active_goals=[],
            now=now,
        )
        assert "rest day — nothing to complete, no action required" in prompt
        assert "not completed" not in prompt


class TestMultipleActivePlans:
    # Non-overlapping active plans can coexist (issue #45); the status prompt must
    # consider all of them, not just one.
    _now = datetime(2025, 6, 4, 8, 0, tzinfo=ZoneInfo("Europe/Helsinki"))  # Wednesday

    def test_multiple_current_plans_all_rendered(self):
        plan_a = _plan(date(2025, 6, 2), name="Endurance Base")
        plan_b = _plan(date(2025, 5, 26), name="Strength Block")
        prompt = _build_status_prompt(
            athlete=_athlete(),
            recent_activities=[],
            current_metric=None,
            active_plans=[
                (plan_a, [_workout(3, workout_type="intervals")]),
                (plan_b, [_workout(3, workout_type="squats")]),
            ],
            active_goals=[],
            now=self._now,
        )
        assert "Active training plan: Endurance Base" in prompt
        assert "Active training plan: Strength Block" in prompt
        # Each plan's own week is rendered (plan_b started a week earlier).
        assert "intervals" in prompt
        assert "squats" in prompt

    def test_upcoming_plan_noted_without_this_week_detail(self):
        upcoming = _plan(date(2025, 7, 1), name="Race Prep", end=date(2025, 8, 15))
        prompt = _build_status_prompt(
            athlete=_athlete(),
            recent_activities=[],
            current_metric=None,
            active_plans=[(upcoming, [])],
            active_goals=[],
            now=self._now,
        )
        assert "Upcoming training plan: Race Prep" in prompt
        assert "2025-07-01" in prompt
        assert "Current week" not in prompt
        assert "This week's planned workouts" not in prompt

    def test_ended_plan_excluded(self):
        ended = _plan(date(2025, 4, 1), name="Old Block", end=date(2025, 5, 1))
        prompt = _build_status_prompt(
            athlete=_athlete(),
            recent_activities=[],
            current_metric=None,
            active_plans=[(ended, [])],
            active_goals=[],
            now=self._now,
        )
        assert "Old Block" not in prompt
        assert "training plan" not in prompt

    def test_mixed_current_upcoming_ended(self):
        current = _plan(date(2025, 6, 2), name="Current Base")
        upcoming = _plan(date(2025, 7, 1), name="Future Prep")
        ended = _plan(date(2025, 4, 1), name="Past Block", end=date(2025, 5, 1))
        prompt = _build_status_prompt(
            athlete=_athlete(),
            recent_activities=[],
            current_metric=None,
            active_plans=[
                (ended, []),
                (current, [_workout(3, workout_type="threshold")]),
                (upcoming, []),
            ],
            active_goals=[],
            now=self._now,
        )
        assert "Active training plan: Current Base" in prompt
        assert "Current week" in prompt
        assert "threshold" in prompt
        assert "Upcoming training plan: Future Prep" in prompt
        assert "Past Block" not in prompt


class TestExperienceLevel:
    def _prompt(self, app_settings):
        now = datetime(2025, 6, 4, 8, 0, tzinfo=ZoneInfo("Europe/Helsinki"))
        return _build_status_prompt(
            athlete=_athlete(app_settings),
            recent_activities=[],
            current_metric=None,
            active_plans=[],
            active_goals=[],
            now=now,
        )

    def test_experience_level_included_when_set(self):
        assert "experience level: elite" in self._prompt({"experience_level": "elite"})

    def test_experience_level_absent_when_unset(self):
        assert "experience level" not in self._prompt({})

    def test_system_prompt_includes_guidance(self):
        assert "experience level" in _build_system_prompt()
