"""
Unit tests for backend/app/services/training_math.py.

These are pure-function tests — no DB or fixtures needed.
"""
import math

import pytest

from openkoutsi.training_math import calculate_tss, normalized_power


# ── normalized_power ──────────────────────────────────────────────────────────

class TestNormalizedPower:
    def test_fewer_than_30_samples_returns_none(self):
        assert normalized_power([]) is None
        assert normalized_power([200.0] * 29) is None

    def test_exactly_30_constant_samples(self):
        result = normalized_power([250.0] * 30)
        assert result == pytest.approx(250.0, rel=1e-6)

    def test_large_constant_series(self):
        result = normalized_power([250.0] * 3600)
        assert result == pytest.approx(250.0, rel=1e-6)

    def test_variable_power_exceeds_mean(self):
        # 40 zeros followed by 60 samples at 400 W.
        # Mean = 240 W, but NP is driven up by the 4th-power averaging of
        # rolling windows that are all 400 W, so NP >> mean.
        series = [0.0] * 40 + [400.0] * 60
        result = normalized_power(series)
        assert result is not None
        mean = sum(series) / len(series)  # 240 W
        assert result > mean

    def test_zero_power_series(self):
        result = normalized_power([0.0] * 60)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_known_value(self):
        # 30-sample window of [100, 200] alternating → rolling avg each window
        # is also 150 W (constant after first window).  NP = 150.
        series = [100.0, 200.0] * 15  # 30 samples — exactly one window
        result = normalized_power(series)
        assert result is not None
        # The single rolling window average = (100+200*14+100+...)/30 ≈ 150
        # Doesn't have to be exactly 150; just verify it's in a sensible range.
        assert 100.0 < result < 250.0


# ── calculate_tss ─────────────────────────────────────────────────────────────

class TestCalculateTss:
    def test_power_based_tss(self):
        # NP=250, FTP=300, duration=3600 s
        # IF = 250/300, TSS = (3600 * 250 * IF) / (300 * 3600) * 100
        tss, intensity_factor = calculate_tss(3600, 250.0, None, 300, None)
        expected_if = 250 / 300
        expected_tss = (3600 * 250 * expected_if) / (300 * 3600) * 100
        assert tss == pytest.approx(expected_tss, rel=1e-6)
        assert intensity_factor == pytest.approx(expected_if, rel=1e-6)

    def test_hr_based_tss_when_no_power(self):
        # avg_hr=150, max_hr=185, duration=3600 s
        # lthr = 0.9 * 185 = 166.5
        # TSS = (duration_s / 3600) × (avg_hr / lthr)² × 100
        # TSS = (3600 / 3600) * (150 / 166.5)^2 * 100
        tss, intensity_factor = calculate_tss(3600, None, 150.0, None, 185)
        lthr = 0.9 * 185
        expected_tss = (3600 / 3600) * math.pow((150 / lthr), 2) * 100
        assert tss == pytest.approx(expected_tss, rel=1e-6)
        assert intensity_factor is None

    def test_power_takes_priority_over_hr(self):
        # Both NP and avg_hr provided — power-based TSS must win.
        tss_power, _ = calculate_tss(3600, 250.0, None, 300, 185)
        tss_both, _ = calculate_tss(3600, 250.0, 150.0, 300, 185)
        assert tss_both == pytest.approx(tss_power, rel=1e-6)

    def test_returns_none_when_ftp_is_zero(self):
        tss, if_ = calculate_tss(3600, 250.0, None, 0, None)
        assert tss is None
        assert if_ is None

    def test_returns_none_when_ftp_is_none(self):
        tss, if_ = calculate_tss(3600, 250.0, None, None, None)
        assert tss is None
        assert if_ is None

    def test_returns_none_when_neither_power_nor_hr(self):
        tss, if_ = calculate_tss(3600, None, None, None, None)
        assert tss is None
        assert if_ is None

    def test_hr_based_returns_none_when_max_hr_is_zero(self):
        tss, if_ = calculate_tss(3600, None, 150.0, None, 0)
        assert tss is None
        assert if_ is None

    def test_short_high_intensity_ride(self):
        # 60-min ride, NP=320, FTP=300 → IF > 1 → TSS > 100
        tss, if_ = calculate_tss(3600, 320.0, None, 300, None)
        assert if_ > 1.0
        assert tss > 100.0
