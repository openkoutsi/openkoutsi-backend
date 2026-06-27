"""
Unit tests for the LLM structured-workout synthesizer.

The LLM HTTP call is never made here — we exercise the pure parse/validate
helpers and the date-mapping logic directly.
"""
import json
from datetime import date

import pytest

from backend.app.services.llm_workout_generator import (
    WorkoutGenerationError,
    _build_user_prompt,
    _parse_steps,
)
from backend.app.api.plans import _planned_date
from backend.app.models.team_orm import PlannedWorkout


def _valid_steps_json() -> str:
    return json.dumps({
        "steps": [
            {"kind": "step", "step_type": "warmup",
             "duration": {"type": "time", "seconds": 600},
             "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 50}}},
            {"kind": "repeat", "repeat_count": 3, "steps": [
                {"kind": "step", "step_type": "active",
                 "duration": {"type": "time", "seconds": 300},
                 "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 105}}},
                {"kind": "step", "step_type": "recovery",
                 "duration": {"type": "time", "seconds": 180},
                 "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 50}}},
            ]},
            {"kind": "step", "step_type": "cooldown",
             "duration": {"type": "time", "seconds": 600}},
        ]
    })


class TestParseSteps:
    def test_valid_response_parses_and_serialises(self):
        steps = _parse_steps(_valid_steps_json())
        assert isinstance(steps, list)
        assert steps[0]["kind"] == "step"
        assert steps[1]["kind"] == "repeat"
        assert steps[1]["repeat_count"] == 3

    def test_strips_markdown_fences(self):
        raw = "```json\n" + _valid_steps_json() + "\n```"
        steps = _parse_steps(raw)
        assert len(steps) == 3

    def test_bare_list_accepted(self):
        bare = json.loads(_valid_steps_json())["steps"]
        steps = _parse_steps(json.dumps(bare))
        assert len(steps) == 3

    def test_invalid_json_raises(self):
        with pytest.raises(WorkoutGenerationError):
            _parse_steps("this is not json at all")

    def test_empty_steps_raises(self):
        with pytest.raises(WorkoutGenerationError):
            _parse_steps(json.dumps({"steps": []}))

    def test_schema_violation_raises(self):
        bad = json.dumps({"steps": [
            {"kind": "step", "step_type": "banana",
             "duration": {"type": "time", "seconds": 600}},
        ]})
        with pytest.raises(WorkoutGenerationError):
            _parse_steps(bad)

    def test_nested_repeat_rejected(self):
        # Depth-1 nesting (a repeat inside a repeat) is allowed, matching
        # workouts._validate_steps; only deeper nesting (depth > 1) is rejected.
        nested = json.dumps({"steps": [
            {"kind": "repeat", "repeat_count": 2, "steps": [
                {"kind": "repeat", "repeat_count": 2, "steps": [
                    {"kind": "repeat", "repeat_count": 2, "steps": [
                        {"kind": "step", "step_type": "active",
                         "duration": {"type": "time", "seconds": 60}},
                    ]},
                ]},
            ]},
        ]})
        with pytest.raises(WorkoutGenerationError, match="nested"):
            _parse_steps(nested)

    def test_single_level_repeat_nesting_allowed(self):
        # A repeat nested one level deep is within the depth-1 limit.
        nested = json.dumps({"steps": [
            {"kind": "repeat", "repeat_count": 2, "steps": [
                {"kind": "repeat", "repeat_count": 2, "steps": [
                    {"kind": "step", "step_type": "active",
                     "duration": {"type": "time", "seconds": 60}},
                ]},
            ]},
        ]})
        steps = _parse_steps(nested)
        assert steps[0]["kind"] == "repeat"


class TestBuildUserPrompt:
    def test_includes_descriptor_fields(self):
        pw = PlannedWorkout(
            plan_id="p", week_number=1, day_of_week=2,
            workout_type="threshold", description="2x20 at FTP",
            duration_min=60, target_tss=80,
        )
        prompt = _build_user_prompt(pw, 280, "Ride")
        assert "threshold" in prompt
        assert "60 minutes" in prompt
        assert "80" in prompt
        assert "280" in prompt
        assert "2x20 at FTP" in prompt


class TestPlannedDate:
    def test_first_day_is_start_date(self):
        start = date(2025, 6, 2)  # Monday
        assert _planned_date(start, 1, 1) == start

    def test_week_and_day_offset(self):
        start = date(2025, 6, 2)
        # Week 2, Wednesday → +7 days +2 days
        assert _planned_date(start, 2, 3) == date(2025, 6, 11)
