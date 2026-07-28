"""Unit tests for the aerobic response metrics (issue #37).

Efficiency factor, variability index, aerobic decoupling and W' balance — all
pure functions in `openkoutsi.training_math`, tested against hand-computed
values and synthetic streams with known injected behaviour.
"""
import pytest

from openkoutsi.training_math import (
    DECOUPLING_MIN_DURATION_S,
    aerobic_decoupling,
    decoupling_unavailable_reason,
    efficiency_factor,
    variability_index,
    w_bal_stream,
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

    def test_all_zero_hr_is_degenerate(self):
        assert decoupling_unavailable_reason(
            7200, [200.0] * 7200, [0.0] * 7200
        ) == "degenerate_hr"

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
