"""Tests for openkoutsi/plan_schema.py structure parameters (issue #29)."""
import pytest

from openkoutsi.plan_schema import (
    DayConfig,
    PlanConfig,
    clamp_plan_params,
    plan_defaults_for,
)


class TestPlanDefaultsFor:
    def test_novice_is_conservative(self):
        d = plan_defaults_for("novice")
        assert d["weekly_progression_pct"] == 5.0
        assert d["build_weeks"] == 2
        assert d["intensity_preference"] == "low"

    def test_experienced_is_aggressive(self):
        d = plan_defaults_for("experienced")
        assert d["weekly_progression_pct"] == 9.0
        assert d["build_weeks"] == 3

    def test_unknown_level_falls_back_to_intermediate(self):
        assert plan_defaults_for(None) == plan_defaults_for("intermediate")
        assert plan_defaults_for("bogus") == plan_defaults_for("intermediate")


class TestClampPlanParams:
    def _config(self, **over):
        base = dict(
            days_per_week=1,
            day_configs=[DayConfig(day_of_week=2, workout_type="threshold")],
        )
        base.update(over)
        return PlanConfig(**base)

    def test_progression_pct_clamped(self):
        clamped = clamp_plan_params(self._config(weekly_progression_pct=50.0))
        assert clamped.weekly_progression_pct == 12.0

    def test_build_weeks_clamped(self):
        clamped = clamp_plan_params(self._config(build_weeks=9))
        assert clamped.build_weeks == 4

    def test_negative_base_load_floored_to_zero(self):
        clamped = clamp_plan_params(self._config(weekly_base_load=-5))
        assert clamped.weekly_base_load == 0

    def test_inverted_hours_range_reordered(self):
        clamped = clamp_plan_params(self._config(weekly_hours_min=8, weekly_hours_max=4))
        assert clamped.weekly_hours_min == 4
        assert clamped.weekly_hours_max == 8

    def test_single_hours_endpoint_becomes_point_value(self):
        clamped = clamp_plan_params(self._config(weekly_hours_min=5))
        assert clamped.weekly_hours_min == 5
        assert clamped.weekly_hours_max == 5

    def test_hours_out_of_bounds_clamped(self):
        clamped = clamp_plan_params(self._config(weekly_hours_max=100))
        assert clamped.weekly_hours_max == 40.0

    def test_unset_hours_stay_none(self):
        clamped = clamp_plan_params(self._config())
        assert clamped.weekly_hours_min is None
        assert clamped.weekly_hours_max is None

    def test_defaults_backfill_old_config(self):
        # An old stored config lacking the new keys deserializes with defaults.
        cfg = PlanConfig(days_per_week=1, day_configs=[DayConfig(day_of_week=2, workout_type="easy")])
        assert cfg.weekly_progression_pct == 7.0
        assert cfg.build_weeks == 3
        assert cfg.weekly_base_load == 0
