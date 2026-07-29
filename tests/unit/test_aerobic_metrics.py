"""Unit tests for the aerobic response metrics (issue #37).

Efficiency factor, variability index, aerobic decoupling and W' balance — all
pure functions in `openkoutsi.training_math`, tested against hand-computed
values and synthetic streams with known injected behaviour.
"""
import pytest

from openkoutsi.training_math import (
    DECOUPLING_MAX_VI,
    DECOUPLING_MIN_DURATION_S,
    aerobic_decoupling,
    cp_wprime_plausible,
    decoupling_unavailable_reason,
    efficiency_factor,
    estimate_cp_wprime,
    variability_index,
    w_bal_stream,
    weighted_power,
)


class TestEfficiencyFactor:
    def test_hand_computed(self):
        # 200 W at 140 bpm -> 1.4286 W per beat
        assert efficiency_factor(200.0, 140.0) == pytest.approx(200 / 140)

    @pytest.mark.parametrize(
        "wp,hr",
        [(None, 140.0), (200.0, None), (None, None), (200.0, 0.0), (0.0, 140.0),
         (200.0, -5.0), (-200.0, 140.0)],
    )
    def test_missing_or_nonpositive_inputs_return_none(self, wp, hr):
        assert efficiency_factor(wp, hr) is None


class TestVariabilityIndex:
    def test_hand_computed(self):
        assert variability_index(220.0, 200.0) == pytest.approx(1.1)

    def test_perfectly_steady_ride_is_one(self):
        assert variability_index(200.0, 200.0) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "wp,avg",
        [(None, 200.0), (200.0, None), (200.0, 0.0), (0.0, 200.0)],
    )
    def test_missing_or_nonpositive_inputs_return_none(self, wp, avg):
        assert variability_index(wp, avg) is None


class TestAerobicDecoupling:
    def test_zero_drift_when_both_halves_identical(self):
        power = [200.0] * 7200
        hr = [140.0] * 7200
        assert aerobic_decoupling(power, hr) == pytest.approx(0.0, abs=1e-9)

    def test_known_injected_hr_drift(self):
        # Constant power; heart rate 10% higher in the second half. The ratio
        # falls by exactly 1 - 1/1.1, i.e. ~9.0909% decoupling.
        power = [200.0] * 7200
        hr = [140.0] * 3600 + [154.0] * 3600
        expected = (1 - 1 / 1.1) * 100
        assert aerobic_decoupling(power, hr) == pytest.approx(expected, rel=1e-6)

    def test_known_injected_power_fade(self):
        # Constant heart rate; power drops 10% in the second half -> same drift.
        power = [200.0] * 3600 + [180.0] * 3600
        hr = [140.0] * 7200
        assert aerobic_decoupling(power, hr) == pytest.approx(10.0, rel=1e-6)

    def test_negative_when_second_half_more_efficient(self):
        power = [200.0] * 3600 + [220.0] * 3600
        hr = [140.0] * 7200
        assert aerobic_decoupling(power, hr) < 0

    def test_odd_length_splits_cleanly(self):
        # 7201 samples: the middle one is dropped so both halves are 3600 long.
        # Making that single middle sample wildly different must not move the
        # result off zero.
        power = [200.0] * 3600 + [9999.0] + [200.0] * 3600
        hr = [140.0] * 7201
        assert aerobic_decoupling(power, hr) == pytest.approx(0.0, abs=1e-9)

    def test_streams_of_unequal_length_are_truncated(self):
        power = [200.0] * 7200
        hr = [140.0] * 3600 + [154.0] * 3600 + [140.0] * 500
        # Truncated to 7200 -> the trailing HR samples are ignored.
        expected = (1 - 1 / 1.1) * 100
        assert aerobic_decoupling(power, hr) == pytest.approx(expected, rel=1e-6)

    def test_short_halves_fall_back_to_mean_power(self):
        # Under the 30-sample weighted-power window; must still produce a number.
        power = [200.0] * 10 + [180.0] * 10
        hr = [140.0] * 20
        assert aerobic_decoupling(power, hr) == pytest.approx(10.0, rel=1e-6)

    @pytest.mark.parametrize(
        "power,hr",
        [
            ([], [140.0] * 100),
            ([200.0] * 100, []),
            ([200.0], [140.0]),          # n // 2 == 0
            ([0.0] * 100, [140.0] * 100),  # no usable power
            ([200.0] * 100, [0.0] * 100),  # no usable heart rate
        ],
    )
    def test_unusable_streams_return_none(self, power, hr):
        assert aerobic_decoupling(power, hr) is None


class TestDecouplingGate:
    def _steady_hour(self):
        return [200.0] * 4000, [140.0 + (i % 7) for i in range(4000)]

    def test_valid_steady_endurance_ride_passes(self):
        power, hr = self._steady_hour()
        assert decoupling_unavailable_reason(
            4000, power, hr, workout_category="endurance", vi=1.03
        ) is None

    def test_too_short(self):
        power, hr = self._steady_hour()
        assert decoupling_unavailable_reason(
            DECOUPLING_MIN_DURATION_S - 1, power, hr, "endurance", 1.03
        ) == "too_short"

    def test_no_power(self):
        assert decoupling_unavailable_reason(7200, [], [140.0] * 7200) == "no_power"
        assert decoupling_unavailable_reason(7200, None, [140.0] * 7200) == "no_power"

    def test_no_hr(self):
        assert decoupling_unavailable_reason(7200, [200.0] * 7200, []) == "no_hr"
        assert decoupling_unavailable_reason(7200, [200.0] * 7200, None) == "no_hr"

    def test_flat_hr_is_degenerate(self):
        assert decoupling_unavailable_reason(
            7200, [200.0] * 7200, [140.0] * 7200
        ) == "degenerate_hr"

    def test_all_zero_hr_reads_as_missing_not_degenerate(self):
        # A strap that recorded nothing at all is absent data, not unusable
        # data — `degenerate_hr` is reserved for a flat but positive trace.
        assert decoupling_unavailable_reason(
            7200, [200.0] * 7200, [0.0] * 7200
        ) == "no_hr"

    @pytest.mark.parametrize("category", ["vo2max", "anaerobic", "sprint"])
    def test_interval_categories_rejected(self, category):
        power, hr = self._steady_hour()
        assert decoupling_unavailable_reason(
            4000, power, hr, category, 1.02
        ) == "variable_effort"

    def test_high_variability_index_rejected(self):
        power, hr = self._steady_hour()
        assert decoupling_unavailable_reason(
            4000, power, hr, "tempo", 1.25
        ) == "variable_effort"

    def test_missing_vi_and_category_still_passes(self):
        power, hr = self._steady_hour()
        assert decoupling_unavailable_reason(4000, power, hr) is None

    def test_stream_checks_take_priority_over_duration(self):
        # A short ride with no power reports the missing stream, not the length,
        # so the athlete is told the actually-blocking problem.
        assert decoupling_unavailable_reason(60, [], []) == "no_power"


class TestReviewRegressions:
    """Cases from the #74 robustness review, using the reported inputs verbatim.

    Each one previously produced a number (or a persisted value) where it should
    have produced a refusal.
    """

    def test_steady_rider_fits_a_negative_w_prime_and_is_rejected(self):
        # The OLS intercept is unconstrained, so a rider whose short bests sit
        # below the work-time line — anyone who only rides steady — fits a
        # negative W'. Previously stored as `cp_w=209.3, w_prime_j=-3638`.
        bests = {120: 190.0, 180: 192.0, 300: 195.0, 480: 198.0, 900: 205.0, 1200: 207.0}
        cp, w_prime = estimate_cp_wprime(bests)
        assert cp == pytest.approx(209.32, abs=0.1)
        assert w_prime < 0
        assert cp_wprime_plausible(cp, w_prime) is False
        assert w_bal_stream([250.0] * 60, cp, w_prime) == []

    def test_tiny_positive_w_prime_is_rejected(self):
        # Worse than the negative case: it cleared the old `w_prime > 0` guard
        # and wrote a curve that cratered to zero within eight seconds.
        bests = {120: 206.0, 180: 204.0, 300: 202.0, 480: 201.0, 900: 200.0, 1200: 200.0}
        cp, w_prime = estimate_cp_wprime(bests)
        assert 0 < w_prime < 1000
        assert cp_wprime_plausible(cp, w_prime) is False
        assert w_bal_stream([300.0] * 20 + [100.0] * 80, cp, w_prime) == []

    def test_a_normal_fit_is_still_accepted(self):
        assert cp_wprime_plausible(250.0, 20_000.0) is True
        assert len(w_bal_stream([300.0] * 60, 250.0, 20_000.0)) == 60

    @pytest.mark.parametrize(
        "cp,w_prime",
        [(49.0, 20_000.0), (601.0, 20_000.0), (250.0, 4_999.0), (250.0, 50_001.0),
         (None, 20_000.0), (250.0, None)],
    )
    def test_out_of_range_pairs_rejected(self, cp, w_prime):
        assert cp_wprime_plausible(cp, w_prime) is False

    def test_dead_power_meter_is_not_reported_as_a_heart_rate_problem(self):
        # A paired-but-silent meter records a full stream of zeros. This used to
        # pass the gate, return None from the math, and get stamped
        # `degenerate_hr` — sending the athlete after the wrong device.
        power = [0.0] * 4000
        hr = [130.0 + (i % 20) for i in range(4000)]
        assert decoupling_unavailable_reason(4000, power, hr, "endurance", None) == "no_power"

    def test_power_meter_dying_halfway_is_caught(self):
        power = [200.0] * 2000 + [0.0] * 2000
        hr = [140.0 + (i % 5) for i in range(4000)]
        assert decoupling_unavailable_reason(4000, power, hr, "endurance", 1.02) == "no_power"

    def test_negative_split_is_gated_despite_a_low_variability_index(self):
        # ~150 W then ~200 W: VI ≈ 1.03, comfortably under the 1.10 threshold,
        # but decoupling came out at −15.7%, which an athlete reads as "my
        # durability improved 15%" when they simply rode the back half harder.
        power = [150.0] * 2000 + [200.0] * 2000
        hr = [130.0 + (i % 3) for i in range(2000)] + [150.0 + (i % 3) for i in range(2000)]
        vi = variability_index(weighted_power(power), sum(power) / len(power))
        assert vi < DECOUPLING_MAX_VI  # the old gate let this through
        assert aerobic_decoupling(power, hr) < -15  # and this is what it produced
        assert decoupling_unavailable_reason(
            4000, power, hr, "endurance", vi
        ) == "uneven_pacing"

    def test_ramp_in_the_other_direction_is_also_gated(self):
        power = [220.0] * 2000 + [160.0] * 2000
        hr = [150.0 + (i % 3) for i in range(4000)]
        vi = variability_index(weighted_power(power), sum(power) / len(power))
        assert decoupling_unavailable_reason(
            4000, power, hr, "endurance", vi
        ) == "uneven_pacing"

    def test_evenly_paced_ride_still_passes(self):
        power = [200.0] * 2000 + [195.0] * 2000  # 2.5% apart, well inside tolerance
        hr = [140.0 + (i % 5) for i in range(4000)]
        vi = variability_index(weighted_power(power), sum(power) / len(power))
        assert decoupling_unavailable_reason(4000, power, hr, "endurance", vi) is None

    def test_misaligned_streams_are_refused(self):
        # The FIT parser appends each channel independently, so a strap dropout
        # shifts HR against power rather than leaving a gap. Pairing them
        # sample-for-sample after that produces a confident wrong answer.
        power = [200.0] * 4000
        hr = [140.0 + (i % 5) for i in range(3000)]
        assert decoupling_unavailable_reason(
            4000, power, hr, "endurance", 1.02
        ) == "stream_mismatch"

    def test_small_length_difference_is_tolerated(self):
        power = [200.0] * 4000
        hr = [140.0 + (i % 5) for i in range(3950)]  # 1.3% — normal trailing trim
        assert decoupling_unavailable_reason(4000, power, hr, "endurance", 1.02) is None

    def test_sparse_recording_fails_the_sample_count_check(self):
        # Four hours elapsed, forty minutes recorded: clears the elapsed-time
        # minimum, then gets split into two twenty-minute halves.
        power = [200.0] * 2400
        hr = [140.0 + (i % 5) for i in range(2400)]
        assert decoupling_unavailable_reason(
            14400, power, hr, "endurance", 1.02
        ) == "too_short"


class TestWBalStream:
    CP = 250.0
    W_PRIME = 20000.0

    def test_constant_power_above_cp_depletes_linearly(self):
        # 50 W above CP -> 50 J spent per second.
        stream = w_bal_stream([300.0] * 100, self.CP, self.W_PRIME)
        assert len(stream) == 100
        assert stream[0] == pytest.approx(self.W_PRIME - 50)
        assert stream[9] == pytest.approx(self.W_PRIME - 500)
        assert stream[99] == pytest.approx(self.W_PRIME - 5000)

    def test_below_cp_reconstitutes_toward_w_prime(self):
        depleted = w_bal_stream([300.0] * 200, self.CP, self.W_PRIME)[-1]
        recovering = w_bal_stream(
            [300.0] * 200 + [100.0] * 600, self.CP, self.W_PRIME
        )
        assert recovering[200] > depleted
        # Never falls back while under CP, ends higher than it started, and
        # never overshoots a full tank.
        tail = recovering[200:]
        assert all(a <= b + 1e-9 for a, b in zip(tail, tail[1:]))
        assert tail[-1] > tail[0]
        assert max(tail) <= self.W_PRIME

    def test_reconstitution_is_exponential_not_linear(self):
        # Recovery slows as the tank refills, which is the whole point of the
        # differential form: the first second back must add more than a later one.
        stream = w_bal_stream(
            [400.0] * 100 + [0.0] * 100, self.CP, self.W_PRIME
        )
        first_gain = stream[100] - stream[99]
        later_gain = stream[150] - stream[149]
        assert first_gain > later_gain > 0

    def test_full_depletion_clamps_at_zero(self):
        # 1000 W for an hour would notionally spend 2.7 MJ against a 20 kJ tank.
        stream = w_bal_stream([1000.0] * 3600, self.CP, self.W_PRIME)
        assert min(stream) == 0.0
        assert all(v >= 0 for v in stream)

    def test_never_exceeds_w_prime(self):
        stream = w_bal_stream([0.0] * 500, self.CP, self.W_PRIME)
        assert max(stream) <= self.W_PRIME

    def test_all_zero_power_is_a_no_op(self):
        stream = w_bal_stream([0.0] * 300, self.CP, self.W_PRIME)
        assert stream == [self.W_PRIME] * 300

    def test_riding_exactly_at_cp_holds_balance(self):
        stream = w_bal_stream([self.CP] * 300, self.CP, self.W_PRIME)
        assert stream == [self.W_PRIME] * 300

    @pytest.mark.parametrize(
        "power,cp,w_prime",
        [
            ([], 250.0, 20000.0),
            ([300.0] * 100, None, 20000.0),
            ([300.0] * 100, 250.0, None),
            ([300.0] * 100, 0.0, 20000.0),
            ([300.0] * 100, -10.0, 20000.0),
            ([300.0] * 100, 250.0, 0.0),
            ([300.0] * 100, 250.0, -1.0),
        ],
    )
    def test_missing_or_invalid_inputs_return_empty(self, power, cp, w_prime):
        assert w_bal_stream(power, cp, w_prime) == []
