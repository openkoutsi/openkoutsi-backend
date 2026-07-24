"""Tests for openkoutsi/plan_builder.py."""
import pytest

from openkoutsi.plan_builder import (
    build_all_week_meta,
    build_week_from_config,
    build_week_meta,
    describe_workout,
    intensity_multiplier,
    is_recovery_week,
    progression_factor,
    progression_scale,
    week_meta_from_weeks,
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

        normal_tss = sum(w["target_load"] or 0 for w in normal_week)
        recovery_tss = sum(w["target_load"] or 0 for w in recovery_week)
        assert recovery_tss < normal_tss

    def test_high_intensity_scales_up_tss(self):
        normal_config = self._config(intensity="moderate")
        high_config = self._config(intensity="high")
        normal_week = build_week_from_config(normal_config, 2, 8)
        high_week = build_week_from_config(high_config, 2, 8)

        normal_tss = sum(w["target_load"] or 0 for w in normal_week)
        high_tss = sum(w["target_load"] or 0 for w in high_week)
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
        recovery_tss = next(w["target_load"] for w in recovery if w["day_of_week"] == 2)
        # yoga is off-bike: fixed prescription, never scaled → raw base Load of 10
        assert recovery_tss == 10


# --- issue #29: structure/progression parameters ----------------------------

class TestRecoveryCadence:
    def test_two_one_cadence_puts_recovery_on_week_three(self):
        # build_weeks=2 → cycle length 3 → week 3 is recovery
        assert is_recovery_week(3, 12, build_weeks=2)
        assert not is_recovery_week(2, 12, build_weeks=2)

    def test_three_one_cadence_puts_recovery_on_week_four(self):
        assert is_recovery_week(4, 12, build_weeks=3)
        assert not is_recovery_week(3, 12, build_weeks=3)

    def test_final_week_is_always_recovery(self):
        assert is_recovery_week(8, 8, build_weeks=3)

    def test_cadence_reflected_in_week_meta(self):
        config = PlanConfig(
            days_per_week=3,
            day_configs=[
                DayConfig(day_of_week=2, workout_type="threshold"),
                DayConfig(day_of_week=4, workout_type="endurance"),
                DayConfig(day_of_week=6, workout_type="long"),
            ],
            build_weeks=2,
        )
        meta = build_all_week_meta(config, 6)
        types = [m["week_type"] for m in meta]
        # 2 build weeks then a recovery week, repeating
        assert types[:3] == ["build", "build", "recovery"]


class TestProgressionBounds:
    def test_weekly_step_matches_configured_pct(self):
        config = PlanConfig(
            days_per_week=1,
            day_configs=[DayConfig(day_of_week=2, workout_type="endurance")],
            build_weeks=3,
            weekly_progression_pct=8.0,
            intensity_preference="moderate",
        )
        # Weeks 1→2 are both build weeks in the first block.
        s1 = progression_scale(1, 12, config)
        s2 = progression_scale(2, 12, config)
        assert s2 / s1 == pytest.approx(1.08, abs=1e-6)

    def test_scale_is_capped_for_long_plans(self):
        config = PlanConfig(
            days_per_week=1,
            day_configs=[DayConfig(day_of_week=2, workout_type="endurance")],
            build_weeks=3,
            weekly_progression_pct=12.0,
            intensity_preference="high",
        )
        scales = [progression_scale(w, 24, config) for w in range(1, 25)]
        assert max(scales) <= intensity_multiplier("high") * 1.6 + 1e-9


class TestWeeklyHoursBand:
    def _config(self, lo, hi):
        return PlanConfig(
            days_per_week=3,
            day_configs=[
                DayConfig(day_of_week=2, workout_type="threshold"),
                DayConfig(day_of_week=4, workout_type="endurance"),
                DayConfig(day_of_week=6, workout_type="long"),
            ],
            build_weeks=3,
            weekly_hours_min=lo,
            weekly_hours_max=hi,
        )

    def test_every_week_total_within_band(self):
        config = self._config(4, 6)
        for w in range(1, 13):
            week = build_week_from_config(config, w, 12)
            hours = sum((d["duration_min"] or 0) for d in week) / 60.0
            assert 4 - 0.3 <= hours <= 6 + 0.3

    def test_recovery_week_near_low_end_peak_near_high_end(self):
        config = self._config(4, 8)
        meta = build_all_week_meta(config, 12)
        build_hours = [m["target_hours"] for m in meta if m["week_type"] == "build"]
        recovery_hours = [m["target_hours"] for m in meta if m["week_type"] == "recovery"]
        assert max(build_hours) >= max(recovery_hours)
        assert min(recovery_hours) <= 5.0

    def test_unset_band_preserves_absolute_scaling(self):
        # Without a band, durations scale off the base params (legacy behaviour).
        config = PlanConfig(
            days_per_week=1,
            day_configs=[DayConfig(day_of_week=6, workout_type="long")],
            build_weeks=3,
        )
        week1 = build_week_from_config(config, 1, 8)
        long_day = next(d for d in week1 if d["day_of_week"] == 6)
        # long base duration is 120 min; week-1 scale is ~1.0 at moderate intensity.
        assert 100 <= long_day["duration_min"] <= 140


class TestConsistentDescriptions:
    def test_threshold_description_reflects_duration(self):
        text = describe_workout("threshold", 75, intensity_preference="high", block_index=1)
        assert "threshold" in text.lower()
        assert "×" in text  # interval breakdown present

    def test_short_and_long_threshold_differ_in_rep_count(self):
        short = describe_workout("threshold", 45, intensity_preference="moderate")
        longer = describe_workout("threshold", 120, intensity_preference="moderate")
        assert short != longer

    @staticmethod
    def _described_minutes(text: str) -> int:
        import re
        m = re.search(
            r"(\d+)×(\d+) min .*?(\d+) min easy between; "
            r"(\d+) min warm-up and (\d+) min cool-down",
            text,
        )
        assert m, f"unexpected interval text: {text!r}"
        reps, rep, rest, warmup, cooldown = map(int, m.groups())
        return warmup + cooldown + reps * rep + max(0, reps - 1) * rest

    def test_short_hard_session_description_fits_duration(self):
        # A short threshold/VO2max day must not describe more minutes than the
        # prescribed duration (regression for review item #2).
        for wtype in ("threshold", "vo2max"):
            for duration in (20, 25, 30, 40, 55, 75):
                text = describe_workout(
                    wtype, duration, intensity_preference="high", block_index=2,
                )
                assert self._described_minutes(text) <= duration, (wtype, duration, text)


class TestHardDayGuardrail:
    def test_back_to_back_hard_days_are_eased(self):
        config = PlanConfig(
            days_per_week=2,
            day_configs=[
                DayConfig(day_of_week=2, workout_type="threshold"),
                DayConfig(day_of_week=3, workout_type="vo2max"),  # adjacent → eased
            ],
            intensity_preference="high",
        )
        week = build_week_from_config(config, 1, 8)
        by_day = {d["day_of_week"]: d for d in week}
        assert by_day[2]["workout_type"] == "threshold"
        assert by_day[3]["workout_type"] == "tempo"  # downgraded

    def test_hard_day_cap_by_intensity(self):
        # low intensity → at most one hard day per week
        config = PlanConfig(
            days_per_week=3,
            day_configs=[
                DayConfig(day_of_week=1, workout_type="threshold"),
                DayConfig(day_of_week=3, workout_type="threshold"),
                DayConfig(day_of_week=5, workout_type="threshold"),
            ],
            intensity_preference="low",
        )
        week = build_week_from_config(config, 1, 8)
        hard = [d for d in week if d["workout_type"] in ("threshold", "vo2max")]
        assert len(hard) == 1


class TestWeekMetaFromWeeks:
    def test_summarises_actual_generated_days(self):
        config = PlanConfig(
            days_per_week=1,
            day_configs=[DayConfig(day_of_week=2, workout_type="endurance")],
            build_weeks=3,
            weekly_base_load=40,
        )
        weeks_data = [
            [{"day_of_week": 2, "workout_type": "endurance",
              "duration_min": 90, "target_load": 70, "description": "x"}],
            [{"day_of_week": 2, "workout_type": "endurance",
              "duration_min": 60, "target_load": 45, "description": "x"}],
        ]
        meta = week_meta_from_weeks(config, weeks_data)
        assert meta[0]["target_load"] == 70
        assert meta[0]["target_hours"] == pytest.approx(1.5)
        assert meta[0]["base_load"] == 40
        assert len(meta) == 2
