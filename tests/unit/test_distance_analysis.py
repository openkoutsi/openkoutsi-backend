"""
Unit tests for best_time_for_distance and compute_distance_bests in training_math.py.
"""
import pytest

from openkoutsi.training_math import (
    DISTANCE_BEST_DISTANCES,
    best_time_for_distance,
    compute_distance_bests,
)


class TestBestTimeForDistance:
    def test_returns_none_for_empty_stream(self):
        assert best_time_for_distance([], 1000) is None

    def test_returns_none_when_total_distance_too_short(self):
        # 5 m/s for 100 s = 500 m total — cannot cover 1 km
        assert best_time_for_distance([5.0] * 100, 1000) is None

    def test_constant_speed_exact_coverage(self):
        # 5 m/s for exactly 200 s = 1000 m
        result = best_time_for_distance([5.0] * 200, 1000)
        assert result == 200

    def test_constant_speed_with_surplus(self):
        # 5 m/s for 300 s = 1500 m — best 1 km window is still 200 s
        result = best_time_for_distance([5.0] * 300, 1000)
        assert result == 200

    def test_picks_fastest_window(self):
        # First 200 s at 5 m/s (1000 m), then 200 s at 10 m/s (2000 m)
        # Best 1 km window is 100 s (at the faster section)
        stream = [5.0] * 200 + [10.0] * 200
        result = best_time_for_distance(stream, 1000)
        assert result == 100

    def test_slow_start_fast_finish(self):
        # 1 m/s for 500 s (500 m), then 10 m/s for 200 s (2000 m)
        # Total 2500 m. Best 1 km window is 100 s (at 10 m/s section)
        stream = [1.0] * 500 + [10.0] * 200
        result = best_time_for_distance(stream, 1000)
        assert result == 100

    def test_zero_speed_segments_handled(self):
        # Stopped for 100 s, then 10 m/s for 200 s
        stream = [0.0] * 100 + [10.0] * 200
        result = best_time_for_distance(stream, 1000)
        assert result == 100

    def test_result_is_integer_seconds(self):
        result = best_time_for_distance([4.0] * 500, 1000)
        assert isinstance(result, int)
        assert result == 250

    def test_minimum_window_not_longer_than_stream(self):
        stream = [3.0] * 500  # 1500 m total
        result = best_time_for_distance(stream, 1000)
        assert result is not None
        assert result <= len(stream)

    def test_exactly_covers_at_end(self):
        # Slow for most, then fast enough at the end — use 210 fast seconds so
        # the window isn't sitting exactly on a floating-point boundary
        stream = [0.1] * 900 + [5.0] * 210  # 1050 m fast, best 1 km window = 200 s
        result = best_time_for_distance(stream, 1000)
        assert result == 200


class TestComputeDistanceBests:
    def test_empty_stream_returns_empty(self):
        assert compute_distance_bests([]) == {}

    def test_short_stream_only_covers_short_distances(self):
        # 5 m/s for 200 s = 1000 m exactly
        bests = compute_distance_bests([5.0] * 200)
        assert 1000 in bests
        assert bests[1000] == 200
        # Distances > 1000 m must be absent
        for d in DISTANCE_BEST_DISTANCES:
            if d > 1000:
                assert d not in bests

    def test_covers_multiple_distances(self):
        # 5 m/s for 2000 s = 10 000 m — should cover 1, 2, 3, 5, 8, 10 km
        bests = compute_distance_bests([5.0] * 2000)
        assert 1000 in bests
        assert 5000 in bests
        assert 10000 in bests
        assert 20000 not in bests

    def test_returns_only_standard_distances(self):
        stream = [5.0] * 5000
        bests = compute_distance_bests(stream)
        for d in bests:
            assert d in DISTANCE_BEST_DISTANCES

    def test_faster_section_wins(self):
        # Two equal-length sections at different speeds
        stream = [3.0] * 2000 + [10.0] * 2000
        bests = compute_distance_bests(stream)
        # Best 1 km at 10 m/s = 100 s; at 3 m/s = 334 s
        assert bests[1000] == 100

    def test_all_values_positive(self):
        stream = [6.0] * 10000
        for d, t in compute_distance_bests(stream).items():
            assert t > 0, f"distance {d} has non-positive time {t}"
