"""
Unit tests for peak_average_power and compute_power_bests in training_math.py.
"""

import pytest

from openkoutsi.training_math import (
    CP_FIT_DURATIONS,
    POWER_BEST_DURATIONS,
    compute_power_bests,
    estimate_cp_wprime,
    estimate_ftp_simple,
    peak_average_power,
)


class TestPeakAveragePower:
    def test_returns_none_when_stream_shorter_than_duration(self):
        assert peak_average_power([], 1) is None
        assert peak_average_power([200.0] * 59, 60) is None

    def test_exactly_one_window_returns_that_average(self):
        stream = [300.0] * 60
        result = peak_average_power(stream, 60)
        assert result == pytest.approx(300.0)

    def test_constant_stream_returns_constant(self):
        for d in [1, 30, 300, 3600]:
            stream = [250.0] * d
            assert peak_average_power(stream, d) == pytest.approx(250.0)

    def test_single_second_returns_max(self):
        stream = [100.0, 400.0, 200.0, 350.0]
        assert peak_average_power(stream, 1) == pytest.approx(400.0)

    def test_picks_highest_window(self):
        # First 5s at 100 W, then 5s at 300 W — best 5s avg should be 300
        stream = [100.0] * 5 + [300.0] * 5
        assert peak_average_power(stream, 5) == pytest.approx(300.0)

    def test_overlapping_windows(self):
        # 10 samples: first 5 at 200, last 5 at 400
        # Best 5s window: [400,400,400,400,400] = 400
        stream = [200.0] * 5 + [400.0] * 5
        assert peak_average_power(stream, 5) == pytest.approx(400.0)

    def test_mixed_signal_best_window(self):
        # Hand-computed: best 3-second window in [100, 200, 300, 400, 100]
        # Windows: [100,200,300]=200, [200,300,400]=300, [300,400,100]=266.7
        # Best = 300
        stream = [100.0, 200.0, 300.0, 400.0, 100.0]
        result = peak_average_power(stream, 3)
        assert result == pytest.approx(300.0)

    def test_zero_power_stream(self):
        stream = [0.0] * 100
        assert peak_average_power(stream, 60) == pytest.approx(0.0)

    def test_duration_equals_stream_length(self):
        stream = [100.0, 200.0, 300.0]
        assert peak_average_power(stream, 3) == pytest.approx(200.0)

    def test_large_stream_performance(self):
        # 3-hour stream (10800 samples) — should complete quickly
        stream = [250.0] * 10800
        result = peak_average_power(stream, 3600)
        assert result == pytest.approx(250.0)


class TestComputePowerBests:
    def test_empty_stream_returns_empty(self):
        assert compute_power_bests([]) == {}

    def test_1s_stream_only_covers_1s_duration(self):
        bests = compute_power_bests([300.0])
        assert 1 in bests
        assert bests[1] == pytest.approx(300.0)
        # All durations > 1s must be absent
        for d in POWER_BEST_DURATIONS:
            if d > 1:
                assert d not in bests

    def test_60s_stream_covers_all_durations_up_to_60s(self):
        stream = [200.0] * 60
        bests = compute_power_bests(stream)
        for d in POWER_BEST_DURATIONS:
            if d <= 60:
                assert d in bests, f"duration {d}s should be present"
            else:
                assert d not in bests, f"duration {d}s should be absent"

    def test_all_values_are_positive(self):
        stream = [150.0 + i * 0.1 for i in range(3600)]
        bests = compute_power_bests(stream)
        for d, v in bests.items():
            assert v > 0, f"duration {d}s has non-positive value {v}"

    def test_longer_durations_not_higher_than_shorter(self):
        # For a constant stream, all durations yield the same value
        stream = [250.0] * 3600
        bests = compute_power_bests(stream)
        values = list(bests.values())
        assert all(abs(v - 250.0) < 1e-6 for v in values)

    def test_returns_only_standard_durations(self):
        stream = [200.0] * 3600
        bests = compute_power_bests(stream)
        for d in bests:
            assert d in POWER_BEST_DURATIONS

    def test_full_8h_stream_covers_all_durations(self):
        stream = [300.0] * 28800
        bests = compute_power_bests(stream)
        assert set(bests.keys()) == set(POWER_BEST_DURATIONS)


class TestEstimateFtpSimple:
    def test_returns_95_percent_of_20min_power(self):
        assert estimate_ftp_simple(300.0) == pytest.approx(285.0)

    def test_returns_none_when_no_power(self):
        assert estimate_ftp_simple(None) is None


class TestEstimateCpWPrime:
    def test_recovers_known_cp_and_wprime(self):
        # Construct bests exactly on the line P(t) = CP + W'/t, so the
        # work–time OLS fit must recover CP and W' exactly.
        cp_true, w_prime_true = 250.0, 15000.0
        bests = {d: cp_true + w_prime_true / d for d in CP_FIT_DURATIONS}
        cp, w_prime = estimate_cp_wprime(bests)
        assert cp == pytest.approx(cp_true)
        assert w_prime == pytest.approx(w_prime_true)

    def test_two_points_exact_fit(self):
        cp_true, w_prime_true = 220.0, 12000.0
        bests = {120: cp_true + w_prime_true / 120, 1200: cp_true + w_prime_true / 1200}
        cp, w_prime = estimate_cp_wprime(bests)
        assert cp == pytest.approx(cp_true)
        assert w_prime == pytest.approx(w_prime_true)

    def test_constant_power_yields_zero_wprime(self):
        bests = {d: 250.0 for d in CP_FIT_DURATIONS}
        cp, w_prime = estimate_cp_wprime(bests)
        assert cp == pytest.approx(250.0)
        assert w_prime == pytest.approx(0.0, abs=1e-6)

    def test_single_point_returns_none(self):
        assert estimate_cp_wprime({1200: 250.0}) == (None, None)

    def test_empty_returns_none(self):
        assert estimate_cp_wprime({}) == (None, None)

    def test_ignores_durations_outside_fit_range(self):
        # 1s and 28800s are not CP fit durations; only the two CP points count.
        cp_true, w_prime_true = 240.0, 10000.0
        bests = {
            1: 900.0,
            300: cp_true + w_prime_true / 300,
            900: cp_true + w_prime_true / 900,
            28800: 180.0,
        }
        cp, w_prime = estimate_cp_wprime(bests)
        assert cp == pytest.approx(cp_true)
        assert w_prime == pytest.approx(w_prime_true)

    def test_non_physical_cp_returns_none(self):
        # Decreasing work with time forces a negative slope (CP <= 0).
        bests = {120: 1000.0, 1200: 50.0}
        cp, w_prime = estimate_cp_wprime(bests)
        assert cp is None and w_prime is None
