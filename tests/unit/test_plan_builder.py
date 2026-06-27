"""Tests for openkoutsi/plan_builder.py."""
import pytest

from openkoutsi.plan_builder import (
    build_week_from_config,
    intensity_multiplier,
    progression_factor,
    week_template,
)
from openkoutsi.plan_schema import DayConfig, PlanConfig


class TestWeekTemplate:
    def test_last_week_is_recovery(self):
        template = week_template(8, 8, None)
        types = {w["workout_type"] for w in template}
        assert "vo2max" not in types
        assert "threshold" not in types

    def test_every_fourth_week_is_recovery(self):
        template = week_template(4, 12, None)
        types = [w["workout_type"] for w in template]
        # recovery week has no threshold or vo2max
        assert "threshold" not in types
        assert "vo2max" not in types

    def test_peak_fitness_near_end_uses_peak_week(self):
        template = week_template(6, 8, "peak_fitness")  # week 6 of 8, >= total-3=5
        types = {w["workout_type"] for w in template}
        assert "vo2max" in types

    def test_normal_week_uses_base_week(self):
        template = week_template(2, 8, None)
        types = [w["workout_type"] for w in template]
        assert "threshold" in types


class TestProgressionFactor:
    def test_recovery_week_returns_07(self):
        assert progression_factor(4, 12, "base_building") == pytest.approx(0.7)

    def test_last_week_returns_07(self):
        assert progression_factor(8, 8, "base_building") == pytest.approx(0.7)

    def test_maintenance_returns_10(self):
        assert progression_factor(2, 8, "maintenance") == pytest.approx(1.0)

    def test_race_prep_final_week_tapers(self):
        factor = progression_factor(7, 8, "race_prep")
        assert factor == pytest.approx(0.75)

    def test_race_prep_midpoint_above_baseline(self):
        factor = progression_factor(3, 8, "race_prep")
        assert factor > 0.85

    def test_base_building_increases_over_time(self):
        early = progression_factor(1, 8, "base_building")
        mid = progression_factor(3, 8, "base_building")
        assert mid > early

    def test_base_building_never_exceeds_11(self):
        factor = progression_factor(7, 8, "base_building")
        assert factor <= 1.1 + 1e-9


class TestIntensityMultiplier:
    def test_low(self):
        assert intensity_multiplier("low") == pytest.approx(0.85)

    def test_moderate(self):
        assert intensity_multiplier("moderate") == pytest.approx(1.0)

    def test_high(self):
        assert intensity_multiplier("high") == pytest.approx(1.15)

    def test_unknown_defaults_to_moderate(self):
        assert intensity_multiplier("unknown") == pytest.approx(1.0)


class TestBuildWeekFromConfig:
    def _config(self, days=None, periodization="base_building", intensity="moderate"):
        days = days or [
            DayConfig(day_of_week=2, workout_type="threshold"),
            DayConfig(day_of_week=4, workout_type="endurance"),
            DayConfig(day_of_week=6, workout_type="long"),
        ]
        return PlanConfig(
            days_per_week=len(days),
            day_configs=days,
            periodization=periodization,
            intensity_preference=intensity,
        )

    def test_returns_seven_days(self):
        config = self._config()
        week = build_week_from_config(config, 1, 8)
        assert len(week) == 7

    def test_configured_days_have_correct_types(self):
        config = self._config()
        week = build_week_from_config(config, 1, 8)
        by_day = {w["day_of_week"]: w for w in week}
        assert by_day[2]["workout_type"] == "threshold"
        assert by_day[4]["workout_type"] == "endurance"
        assert by_day[6]["workout_type"] == "long"

    def test_unconfigured_days_are_rest(self):
        config = self._config()
        week = build_week_from_config(config, 1, 8)
        by_day = {w["day_of_week"]: w for w in week}
        assert by_day[1]["workout_type"] == "rest"
        assert by_day[3]["workout_type"] == "rest"

    def test_recovery_week_scales_down_tss(self):
        config = self._config()
        normal_week = build_week_from_config(config, 1, 8)
        recovery_week = build_week_from_config(config, 4, 8)  # week 4 = recovery

        normal_tss = sum(w["target_tss"] or 0 for w in normal_week)
        recovery_tss = sum(w["target_tss"] or 0 for w in recovery_week)
        assert recovery_tss < normal_tss

    def test_high_intensity_scales_up_tss(self):
        normal_config = self._config(intensity="moderate")
        high_config = self._config(intensity="high")
        normal_week = build_week_from_config(normal_config, 2, 8)
        high_week = build_week_from_config(high_config, 2, 8)

        normal_tss = sum(w["target_tss"] or 0 for w in normal_week)
        high_tss = sum(w["target_tss"] or 0 for w in high_week)
        assert high_tss > normal_tss

    def test_custom_notes_override_description(self):
        days = [DayConfig(day_of_week=2, workout_type="threshold", notes="Custom note")]
        config = PlanConfig(days_per_week=1, day_configs=days, periodization="base_building",
                            intensity_preference="moderate")
        week = build_week_from_config(config, 1, 8)
        by_day = {w["day_of_week"]: w for w in week}
        assert by_day[2]["description"] == "Custom note"

    def test_yoga_on_recovery_week_uses_raw_base_tss(self):
        days = [DayConfig(day_of_week=2, workout_type="yoga")]
        config = PlanConfig(days_per_week=1, day_configs=days, periodization="base_building",
                            intensity_preference="moderate")
        recovery = build_week_from_config(config, 4, 8)
        recovery_tss = next(w["target_tss"] for w in recovery if w["day_of_week"] == 2)
        # yoga base TSS is 10; recovery week skips scaling and uses raw base
        assert recovery_tss == 10
