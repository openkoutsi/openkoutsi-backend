"""Unit tests for the deterministic plan-adherence scoring math (issue #26)."""
import pytest

from openkoutsi.plan_adherence import (
    cycling_match_score,
    deviation_score,
    forgiveness_factor,
    meets_threshold,
    plan_adherence,
    supplemental_match_score,
    supplemental_weight,
)
from openkoutsi.sport_matching import workout_is_cycling


class TestDeviationScore:
    def test_on_target_is_one(self):
        assert deviation_score(100, 100) == 1.0

    def test_symmetric_under_and_over(self):
        # 20% either way → 0.8
        assert deviation_score(80, 100) == pytest.approx(0.8)
        assert deviation_score(120, 100) == pytest.approx(0.8)

    def test_double_or_zero_is_zero(self):
        assert deviation_score(200, 100) == 0.0
        assert deviation_score(0, 100) == 0.0

    def test_missing_target_is_one(self):
        assert deviation_score(50, 0) == 1.0


class TestCyclingMatchScore:
    def test_on_target_both_dims_is_100(self):
        assert cycling_match_score(100, 3600, 100, 60) == pytest.approx(100.0)

    def test_blend_weights_load_more_than_duration(self):
        # Load perfect, duration 50% under → 0.7*1 + 0.3*0.5 = 0.85
        score = cycling_match_score(100, 1800, 100, 60)
        assert score == pytest.approx(85.0)

    def test_over_performing_penalised(self):
        # Both dims 20% over → 0.8 each → 80
        score = cycling_match_score(120, 4320, 100, 60)
        assert score == pytest.approx(80.0)

    def test_load_only_fallback(self):
        assert cycling_match_score(80, 0, 100, None) == pytest.approx(80.0)

    def test_duration_only_fallback(self):
        assert cycling_match_score(0, 1800, None, 60) == pytest.approx(50.0)

    def test_no_targets_completion_only(self):
        assert cycling_match_score(0, 0, None, None) == 100.0

    def test_wild_overshoot_is_raw_zero(self):
        # The pure math is unfloored — the 50 completion floor is applied by the
        # service layer (workout_match_score), not here.
        assert cycling_match_score(300, 8 * 3600, 75, 85) == pytest.approx(0.0)


class TestSupplemental:
    def test_done_is_100_missed_is_0(self):
        assert supplemental_match_score(True) == 100.0
        assert supplemental_match_score(False) == 0.0

    def test_weight_from_mean_cycling_load(self):
        assert supplemental_weight([100, 200]) == pytest.approx(0.75 * 150)

    def test_weight_fallback_when_no_cycling_loads(self):
        assert supplemental_weight([]) == 30.0


class TestForgivenessFactor:
    def test_known_reasons(self):
        assert forgiveness_factor("illness") == 0.90
        assert forgiveness_factor("injury") == 0.90
        assert forgiveness_factor("fatigue") == 0.60
        assert forgiveness_factor("travel") == 0.50
        assert forgiveness_factor("weather") == 0.40

    def test_case_insensitive(self):
        assert forgiveness_factor("Illness") == 0.90

    def test_freeform_and_none_are_discretionary(self):
        assert forgiveness_factor("too lazy") == 0.10
        assert forgiveness_factor("") == 0.10
        assert forgiveness_factor(None) == 0.10


class TestMeetsThreshold:
    def test_at_60pct_passes(self):
        assert meets_threshold(60, 100) is True

    def test_below_60pct_fails(self):
        assert meets_threshold(59, 100) is False

    def test_missing_target_passes(self):
        assert meets_threshold(0, None) is True
        assert meets_threshold(None, 0) is True


class TestPlanAdherence:
    def test_all_on_target_is_100(self):
        assert plan_adherence([(100, 100.0), (50, 100.0)]) == pytest.approx(100.0)

    def test_missed_key_session_hurts_more(self):
        # A big missed session (weight 200, score 0) vs a small perfect one.
        score = plan_adherence([(200, 0.0), (50, 100.0)])
        assert score == pytest.approx(20.0)

    def test_empty_is_none(self):
        assert plan_adherence([]) is None
        assert plan_adherence([(0, 100.0)]) is None


class TestWorkoutIsCycling:
    def test_generic_types_are_cycling(self):
        assert workout_is_cycling("threshold") is True
        assert workout_is_cycling("endurance") is True

    def test_explicit_cycling(self):
        assert workout_is_cycling("ride") is True
        assert workout_is_cycling("bike") is True

    def test_supplemental_sports(self):
        assert workout_is_cycling("strength") is False
        assert workout_is_cycling("yoga") is False
        assert workout_is_cycling("swim") is False
        assert workout_is_cycling("run") is False
