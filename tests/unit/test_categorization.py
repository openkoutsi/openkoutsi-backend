"""Unit tests for openkoutsi/categorization.py."""
import pytest

from openkoutsi.categorization import WorkoutCategory, classify_workout


class TestClassifyWorkout:
    def test_returns_none_without_intensity_factor(self):
        assert classify_workout(None, None) is None
        assert classify_workout(None, 1.05) is None

    def test_recovery(self):
        assert classify_workout(0.60, 1.0) == WorkoutCategory.recovery
        assert classify_workout(0.64, 1.0) == WorkoutCategory.recovery

    def test_endurance(self):
        assert classify_workout(0.65, 1.0) == WorkoutCategory.endurance
        assert classify_workout(0.75, 1.0) == WorkoutCategory.endurance
        assert classify_workout(0.77, 1.0) == WorkoutCategory.endurance

    def test_tempo(self):
        assert classify_workout(0.78, 1.0) == WorkoutCategory.tempo
        assert classify_workout(0.85, 1.0) == WorkoutCategory.tempo
        assert classify_workout(0.89, 1.0) == WorkoutCategory.tempo

    def test_tempo_with_high_vi_upgrades_to_threshold(self):
        assert classify_workout(0.83, 1.15) == WorkoutCategory.threshold

    def test_threshold(self):
        assert classify_workout(0.90, 1.0) == WorkoutCategory.threshold
        assert classify_workout(0.95, 1.0) == WorkoutCategory.threshold
        assert classify_workout(0.99, 1.0) == WorkoutCategory.threshold

    def test_threshold_upgrades_to_vo2max_with_high_vi(self):
        assert classify_workout(0.95, 1.12) == WorkoutCategory.vo2max

    def test_vo2max(self):
        assert classify_workout(1.00, 1.0) == WorkoutCategory.vo2max
        assert classify_workout(1.05, 1.0) == WorkoutCategory.vo2max
        assert classify_workout(1.09, 1.0) == WorkoutCategory.vo2max

    def test_anaerobic(self):
        assert classify_workout(1.10, 1.0) == WorkoutCategory.anaerobic
        assert classify_workout(1.15, 1.0) == WorkoutCategory.anaerobic
        assert classify_workout(1.19, 1.0) == WorkoutCategory.anaerobic

    def test_sprint_from_very_high_if(self):
        assert classify_workout(1.20, 1.0) == WorkoutCategory.sprint
        assert classify_workout(1.50, 1.0) == WorkoutCategory.sprint

    def test_variability_index_none_is_treated_as_one(self):
        # Should not raise; should behave same as VI=1.0
        assert classify_workout(0.70, None) == WorkoutCategory.endurance

    def test_boundary_values(self):
        # Exact boundary at 0.78: endurance/tempo boundary
        assert classify_workout(0.78, 1.0) == WorkoutCategory.tempo
        # Exact boundary at 0.65: recovery/endurance boundary
        assert classify_workout(0.65, 1.0) == WorkoutCategory.endurance
