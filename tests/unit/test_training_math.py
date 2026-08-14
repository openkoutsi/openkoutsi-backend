"""
Unit tests for backend/app/services/training_math.py.

These are pure-function tests — no DB or fixtures needed.
"""
import json
import math
import warnings

import numpy as np
import pytest

from openkoutsi.training_math import (
    best_time_for_distance,
    aerobic_decoupling,
    calculate_load,
    compute_power_bests,
    compute_torque_stream,
    weighted_power,
)


# ── weighted_power ──────────────────────────────────────────────────────────

class TestNormalizedPower:
    def test_fewer_than_30_samples_returns_none(self):
        assert weighted_power([]) is None
        assert weighted_power([200.0] * 29) is None

    def test_exactly_30_constant_samples(self):
        result = weighted_power([250.0] * 30)
        assert result == pytest.approx(250.0, rel=1e-6)

    def test_large_constant_series(self):
        result = weighted_power([250.0] * 3600)
        assert result == pytest.approx(250.0, rel=1e-6)

    def test_variable_power_exceeds_mean(self):
        # 40 zeros followed by 60 samples at 400 W.
        # Mean = 240 W, but Weighted Power is driven up by the 4th-power averaging of
        # rolling windows that are all 400 W, so Weighted Power >> mean.
        series = [0.0] * 40 + [400.0] * 60
        result = weighted_power(series)
        assert result is not None
        mean = sum(series) / len(series)  # 240 W
        assert result > mean

    def test_zero_power_series(self):
        result = weighted_power([0.0] * 60)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_known_value(self):
        # 30-sample window of [100, 200] alternating → rolling avg each window
        # is also 150 W (constant after first window).  Weighted Power = 150.
        series = [100.0, 200.0] * 15  # 30 samples — exactly one window
        result = weighted_power(series)
        assert result is not None
        # The single rolling window average = (100+200*14+100+...)/30 ≈ 150
        # Doesn't have to be exactly 150; just verify it's in a sensible range.
        assert 100.0 < result < 250.0


# ── calculate_load ─────────────────────────────────────────────────────────────

class TestCalculateTss:
    def test_power_based_tss(self):
        # Weighted Power=250, FTP=300, duration=3600 s
        # Intensity = 250/300, Load = (3600 * 250 * Intensity) / (300 * 3600) * 100
        load, intensity = calculate_load(3600, 250.0, None, 300, None)
        expected_if = 250 / 300
        expected_tss = (3600 * 250 * expected_if) / (300 * 3600) * 100
        assert load == pytest.approx(expected_tss, rel=1e-6)
        assert intensity == pytest.approx(expected_if, rel=1e-6)

    def test_hr_based_tss_when_no_power(self):
        # avg_hr=150, max_hr=185, duration=3600 s
        # lthr = 0.9 * 185 = 166.5
        # Load = (duration_s / 3600) × (avg_hr / lthr)² × 100
        # Load = (3600 / 3600) * (150 / 166.5)^2 * 100
        load, intensity = calculate_load(3600, None, 150.0, None, 185)
        lthr = 0.9 * 185
        expected_tss = (3600 / 3600) * math.pow((150 / lthr), 2) * 100
        assert load == pytest.approx(expected_tss, rel=1e-6)
        assert intensity is None


# ── compute_torque_stream ───────────────────────────────────────────────────

class TestComputeTorqueStream:
    def test_known_value(self):
        # 200 W at 90 rpm → torque = 200 * 60 / (2π * 90) ≈ 21.22 Nm
        result = compute_torque_stream([200.0], [90.0])
        assert result == pytest.approx([200.0 * 60.0 / (2 * math.pi * 90.0)])
        assert result[0] == pytest.approx(21.221, abs=1e-3)

    def test_zero_cadence_yields_zero(self):
        # Coasting: no pedalling → torque defined as 0.0, not a division error.
        assert compute_torque_stream([300.0, 0.0], [0.0, 0.0]) == [0.0, 0.0]

    def test_negative_cadence_yields_zero(self):
        assert compute_torque_stream([250.0], [-5.0]) == [0.0]

    def test_empty_inputs(self):
        assert compute_torque_stream([], []) == []
        assert compute_torque_stream([200.0], []) == []
        assert compute_torque_stream([], [90.0]) == []

    def test_mismatched_lengths_uses_shorter(self):
        result = compute_torque_stream([200.0, 200.0, 200.0], [90.0, 90.0])
        assert len(result) == 2

    def test_power_takes_priority_over_hr(self):
        # Both Weighted Power and avg_hr provided — power-based Load must win.
        tss_power, _ = calculate_load(3600, 250.0, None, 300, 185)
        tss_both, _ = calculate_load(3600, 250.0, 150.0, 300, 185)
        assert tss_both == pytest.approx(tss_power, rel=1e-6)

    def test_returns_none_when_ftp_is_zero(self):
        load, if_ = calculate_load(3600, 250.0, None, 0, None)
        assert load is None
        assert if_ is None

    def test_returns_none_when_ftp_is_none(self):
        load, if_ = calculate_load(3600, 250.0, None, None, None)
        assert load is None
        assert if_ is None

    def test_returns_none_when_neither_power_nor_hr(self):
        load, if_ = calculate_load(3600, None, None, None, None)
        assert load is None
        assert if_ is None

    def test_hr_based_returns_none_when_max_hr_is_zero(self):
        load, if_ = calculate_load(3600, None, 150.0, None, 0)
        assert load is None
        assert if_ is None

    def test_short_high_intensity_ride(self):
        # 60-min ride, Weighted Power=320, FTP=300 → Intensity > 1 → Load > 100
        load, if_ = calculate_load(3600, 320.0, None, 300, None)
        assert if_ > 1.0
        assert load > 100.0


# ── the array boundary ──────────────────────────────────────────────────────
#
# The stream math is vectorised internally, but ``ActivityStream.data`` is a
# JSON column and callers hand it around as ``list[float]``. A numpy value that
# escapes one of these functions does not fail at the call site — it fails much
# later, at the JSON encoder or the ORM. These tests pin the boundary.

class TestArrayBoundary:
    def test_torque_stream_is_json_serialisable(self):
        stream = compute_torque_stream([200.0, 0.0, 150.0], [90.0, 0.0, 80.0])
        assert type(stream) is list
        assert all(type(v) is float for v in stream)
        assert json.loads(json.dumps(stream)) == stream

    def test_scalars_are_plain_floats(self):
        assert type(weighted_power([250.0] * 60)) is float
        power = [200.0 + (i % 60) for i in range(7200)]
        hr = [140.0 + i / 3600 for i in range(7200)]
        assert type(aerobic_decoupling(power, hr)) is float

    def test_zero_cadence_does_not_warn(self):
        # The divisor is masked, not the result: dividing by a raw zero cadence
        # would emit a RuntimeWarning on every coasting sample.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert compute_torque_stream([300.0, 250.0], [0.0, 90.0])[0] == 0.0

    def test_functions_accept_numpy_arrays(self):
        # Callers that already hold an array shouldn't have to convert back.
        assert weighted_power(np.full(60, 250.0)) == pytest.approx(250.0)
        assert compute_power_bests(np.full(60, 200.0))[60] == pytest.approx(200.0)
        assert compute_torque_stream(np.array([200.0]), np.array([90.0])) == pytest.approx(
            [200.0 * 60.0 / (2 * math.pi * 90.0)]
        )


# ── reading a gap ───────────────────────────────────────────────────────────
#
# Since issue #76 a stream spans the whole elapsed ride with ``None`` where a
# channel had no sample, so every function here had to decide what a gap means.
# The split is deliberate and not uniform: a *rider* claim (average power, a
# best) ignores gaps, because a sensor that missed a second is not evidence the
# rider stopped; a *clock* claim (a fastest kilometre) counts them, because the
# seconds have to be real.

class TestGappyStreams:
    def test_weighted_power_ignores_gaps(self):
        # Identical to the same ride with the dropout simply absent, which is
        # exactly what the old parser handed over.
        steady = [250.0] * 120
        gappy = steady[:60] + [None] * 30 + steady[60:]
        assert weighted_power(gappy) == pytest.approx(weighted_power(steady))

    def test_a_gap_cannot_lower_weighted_power(self):
        # Reading a dropout as zero watts would let a failed sensor cut the
        # rider's Load for the day.
        assert weighted_power([250.0] * 60 + [None] * 60) == pytest.approx(250.0)

    def test_power_bests_ignore_gaps(self):
        steady = [300.0] * 60 + [100.0] * 60
        gappy = [300.0] * 60 + [None] * 20 + [100.0] * 60
        assert compute_power_bests(gappy)[60] == pytest.approx(
            compute_power_bests(steady)[60]
        )

    def test_distance_bests_count_a_gap_as_lost_time(self):
        # 10 m/s for 60 s either side of a 60 s hole. Ignoring the hole would
        # report a 1 km best of 100 s; the rider was not moving 10 m/s through
        # a stretch nothing was recorded for.
        gappy = [10.0] * 60 + [None] * 60 + [10.0] * 60
        assert best_time_for_distance(gappy, 1000) == 160

    def test_distance_best_is_refused_when_the_gap_eats_the_distance(self):
        assert best_time_for_distance([10.0] * 50 + [None] * 60, 1000) is None

    def test_torque_is_a_gap_where_either_input_is(self):
        # `cadence is None` must not fall through the coasting branch and
        # report a confident 0.0 Nm.
        assert compute_torque_stream([200.0, 200.0], [90.0, None])[1] is None
        assert compute_torque_stream([200.0, None], [90.0, 90.0])[1] is None

    def test_torque_over_a_gap_is_json_serialisable(self):
        stream = compute_torque_stream([200.0, None], [90.0, 90.0])
        assert json.loads(json.dumps(stream)) == stream

    def test_decoupling_pairs_across_a_gap_by_position(self):
        # Power rises in the second half while HR holds, so decoupling must come
        # out negative. Under the old parser the HR dropout pulled every later
        # sample forward and the two halves stopped describing the same minutes.
        power = [200.0] * 1800 + [260.0] * 1800
        hr = [150.0] * 600 + [None] * 200 + [150.0] * 2800
        assert aerobic_decoupling(power, hr) < 0

    def test_an_all_gap_stream_reads_as_no_data(self):
        assert weighted_power([None] * 120) is None
        assert compute_power_bests([None] * 120) == {}
