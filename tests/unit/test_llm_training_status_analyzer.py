"""Unit tests for the LLM training status prompt builder."""
from datetime import date, datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

from backend.app.models.team_orm import Athlete, PlannedWorkout, TrainingPlan
from backend.app.services.llm_training_status_analyzer import _build_status_prompt


def _athlete() -> Athlete:
    a = MagicMock(spec=Athlete)
    a.ftp = 250
    a.max_hr = 190
    return a


def _plan(start: date) -> TrainingPlan:
    p = MagicMock(spec=TrainingPlan)
    p.name = "Base build"
    p.start_date = start
    p.end_date = None
    return p


def _workout(day_of_week: int, **kwargs) -> PlannedWorkout:
    w = MagicMock(spec=PlannedWorkout)
    w.day_of_week = day_of_week
    w.workout_type = kwargs.get("workout_type", "endurance")
    w.target_tss = kwargs.get("target_tss", None)
    w.completed_activity_id = kwargs.get("completed_activity_id", None)
    return w


class TestThisWeekWeekdayLabels:
    def test_workouts_labelled_with_weekday_and_date(self):
        # Plan starts Monday 2025-06-02; "now" is Wednesday 2025-06-04.
        plan_start = date(2025, 6, 2)
        now = datetime(2025, 6, 4, 8, 0, tzinfo=ZoneInfo("Europe/Helsinki"))
        workouts = [
            _workout(1, workout_type="threshold", target_tss=80),  # Monday
            _workout(3, workout_type="intervals", target_tss=90),  # Wednesday = today
            _workout(7, workout_type="long", target_tss=120),       # Sunday
        ]
        prompt = _build_status_prompt(
            athlete=_athlete(),
            recent_activities=[],
            current_metric=None,
            active_plan=_plan(plan_start),
            this_week_workouts=workouts,
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
            active_plan=_plan(plan_start),
            this_week_workouts=workouts,
            active_goals=[],
            now=now,
        )
        assert "Friday 2025-06-06 (today)" in prompt
        # Tuesday falls on the second-to-last day of the Thu..Wed block.
        assert "Tuesday 2025-06-10" in prompt
