"""Unit tests for the aerobic response metrics (issue #37).

Efficiency factor, variability index, aerobic decoupling and W' balance — all
pure functions in `openkoutsi.training_math`, tested against hand-computed
values and synthetic streams with known injected behaviour.
"""
import pytest

from backend.app.services.aerobic_metrics import _sampling_supports_integration
from openkoutsi.training_math import (
    DECOUPLING_MAX_BRIDGED_PAUSE_S,
    DECOUPLING_MAX_VI,
    DECOUPLING_MIN_DURATION_S,
    aerobic_decoupling,
    analyse_decoupling,
    cp_wprime_plausible,
    decoupling_unavailable_reason,
    decoupling_window,
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


class TestGappyStreams:
    """Gaps make the misalignment measurable rather than invisible (issue #76).

    The #74 guard compared the two streams' lengths, which is the only symptom a
    dropout had while the parser appended each channel independently. It caught
    one large dropout and missed the case that motivated #76: two channels each
    losing a similar number of records at different points, ending up the same
    length while being internally misaligned. On streams that share a clock the
    overlap can simply be counted.
    """

    def test_interleaved_dropouts_of_equal_size_are_refused(self):
        # Equal lengths, equal sample counts — invisible to a length check.
        power = [200.0] * 1000 + [None] * 400 + [200.0] * 2600
        hr = [150.0] * 2600 + [None] * 400 + [150.0] * 1000
        assert len(power) == len(hr)
        assert decoupling_unavailable_reason(
            4000, power, hr, "endurance", 1.02
        ) == "stream_mismatch"

    def test_a_gap_shared_by_both_channels_is_not_a_mismatch(self):
        # A device pause leaves the same hole in every channel. The two still
        # describe the same seconds, so the pairing is sound.
        power = [200.0] * 2000 + [None] * 200 + [200.0] * 2000
        hr = [150.0 + (i % 5) for i in range(2000)] + [None] * 200 + [150.0] * 2000
        assert decoupling_unavailable_reason(4200, power, hr, "endurance", 1.02) is None

    def test_a_small_shared_gap_stays_inside_tolerance(self):
        power = [200.0] * 3950 + [None] * 50
        hr = [150.0 + (i % 5) for i in range(3950)] + [None] * 50
        assert decoupling_unavailable_reason(4000, power, hr, "endurance", 1.02) is None

    def test_paired_seconds_are_counted_not_the_width_of_the_grid(self):
        # Four hours elapsed, forty minutes recorded — the halves would be
        # mostly gap. Sparse but *aligned*, so it is short rather than mismatched.
        power = ([200.0] + [None] * 5) * 2400
        hr = ([150.0] + [None] * 5) * 2400
        assert decoupling_unavailable_reason(
            14400, power, hr, "endurance", 1.02
        ) == "too_short"


class TestSamplingGuard:
    """``w_bal`` needs 1 Hz, and gaps make that checkable rather than inferred."""

    def test_a_dense_grid_supports_integration(self):
        assert _sampling_supports_integration([200.0] * 3600, 3600) is True

    def test_a_low_rate_recording_is_rejected(self):
        # One reading every four seconds: the depletion rate would be wrong by
        # the sampling ratio. Before #76 this was indistinguishable from a 1 Hz
        # ride with a dropout, because only the list length was visible.
        smart = ([200.0] + [None] * 3) * 900
        assert _sampling_supports_integration(smart, 3600) is False

    def test_a_ride_with_cafe_stops_still_passes(self):
        # Deliberately loose: a genuine 1 Hz ride that stops for a while has
        # fewer readings than elapsed seconds and must keep the feature.
        with_stops = [200.0] * 2400 + [None] * 1200
        assert _sampling_supports_integration(with_stops, 2400) is True

    def test_pre_issue_76_dense_streams_are_judged_as_before(self):
        # No gaps to count, so this falls back to length vs. duration_s.
        assert _sampling_supports_integration([200.0] * 2000, 3600) is True
        assert _sampling_supports_integration([200.0] * 900, 3600) is False

    def test_an_empty_stream_cannot_be_integrated(self):
        assert _sampling_supports_integration([], 3600) is False


class TestWBalOverGaps:
    def test_a_gap_is_a_second_of_recovery(self):
        # The integration is over the clock, so a second with no reading has to
        # be a second — and a rider not measurably above CP is not spending W'.
        cp, w_prime = 250.0, 20000.0
        drained = w_bal_stream([400.0] * 100, cp, w_prime)
        recovered = w_bal_stream([400.0] * 100 + [None] * 200, cp, w_prime)
        assert len(recovered) == 300
        assert recovered[99] == pytest.approx(drained[-1])
        assert recovered[-1] > recovered[99]

    def test_the_stream_spans_the_whole_grid(self):
        out = w_bal_stream([200.0, None, 200.0], 250.0, 20000.0)
        assert len(out) == 3
        assert all(isinstance(v, float) for v in out)


def _long_ride(
    total_s: int = 7 * 3600, *, stops: list[tuple[int, int]] | None = None, mode: str = "shared"
) -> tuple[list[float | None], list[float | None]]:
    """A steady seven-hour endurance ride with stops cut into it.

    ``mode`` is what the stops look like on the streams:

    ``shared``      no record at all, a hole in every channel — a device pause.
    ``power_only``  heart rate keeps logging while the power meter, which sleeps
                    when the cranks stop, records nothing. This is what a stop on
                    a head unit with auto-pause off actually looks like, and the
                    shape that used to cost the whole ride its figure.
    ``hr_only``     the reverse, a strap that drops out — a fault, not a stop.

    Heart rate drifts up 3% across the ride, so a figure computed over any
    honest window of it lands a little above zero.
    """
    stops = stops or []
    power: list[float | None] = []
    hr: list[float | None] = []
    for i in range(total_s):
        beat = 140.0 * (1 + 0.03 * i / total_s) + (i % 5) * 0.1
        watts = 200.0 + (i % 7)
        stopped = any(start <= i < start + length for start, length in stops)
        if stopped and mode == "shared":
            power.append(None)
            hr.append(None)
        elif stopped and mode == "power_only":
            power.append(None)
            hr.append(round(beat * 0.72, 1))
        elif stopped and mode == "hr_only":
            power.append(watts)
            hr.append(None)
        else:
            power.append(watts)
            hr.append(round(beat, 1))
    return power, hr


class TestDecouplingWindow:
    """The figure is measured over the longest continuous block, not the ride."""

    def test_an_unbroken_ride_is_one_window(self):
        power, hr = _long_ride(4000)
        assert decoupling_window(power, hr) == (0, 4000)

    def test_a_short_stop_does_not_break_the_ride(self):
        # The complaint this fixes: a five-minute stop in a seven-hour ride.
        power, hr = _long_ride(stops=[(3 * 3600, 300)])
        assert decoupling_window(power, hr) == (0, 7 * 3600)

    def test_a_stop_at_the_tolerance_still_bridges(self):
        power, hr = _long_ride(stops=[(3 * 3600, DECOUPLING_MAX_BRIDGED_PAUSE_S)])
        assert decoupling_window(power, hr) == (0, 7 * 3600)

    def test_a_long_stop_splits_the_ride_and_the_longer_side_wins(self):
        # Two hours, an hour for lunch, then four: after an hour off the bike
        # the two sides are separate efforts, so the longer one is measured.
        power, hr = _long_ride(stops=[(2 * 3600, 3600)])
        assert decoupling_window(power, hr) == (3 * 3600, 7 * 3600)

    def test_the_block_is_chosen_by_usable_seconds_not_width(self):
        # Three hours of five-minute efforts between ten-minute stops — one wide
        # block holding an hour of riding — then a solid 4000 s. The wide one
        # covers nearly three times the clock and has less to say, so the
        # narrow one is the window.
        stuttering = ([200.0] * 300 + [None] * 600) * 12
        power = stuttering + [None] * 1200 + [200.0] * 4000
        hr = (
            ([140.0] * 300 + [None] * 600) * 12 + [None] * 1200 + [140.0] * 4000
        )
        start, end = decoupling_window(power, hr)
        assert (start, end) == (len(stuttering) + 1200, len(power))
        assert end - start < len(stuttering)  # narrower, and still the winner

    def test_a_stop_the_power_meter_slept_through_breaks_the_block_too(self):
        # Only heart rate recorded, so there is nothing to pair: for a metric
        # that multiplies one channel against the other this is a hole.
        power, hr = _long_ride(stops=[(2 * 3600, 3600)], mode="power_only")
        assert decoupling_window(power, hr) == (3 * 3600, 7 * 3600)

    def test_no_overlap_at_all_has_no_window(self):
        power = [200.0] * 100 + [None] * 100
        hr = [None] * 100 + [140.0] * 100
        assert decoupling_window(power, hr) is None

    def test_empty_streams_have_no_window(self):
        assert decoupling_window([], []) is None
        assert decoupling_window(None, None) is None


class TestDecouplingOverPauses:
    """The reported bug: a seven-hour ride refused over the stops in it.

    Every one of these produced `stream_mismatch` — "the recordings don't line
    up closely enough to compare them" — because seconds where the power meter
    had stopped broadcasting were counted as heart rate the pairing had failed
    to match. They are stops, not faults.
    """

    def test_a_five_minute_stop_no_longer_costs_the_ride_its_figure(self):
        power, hr = _long_ride(stops=[(3 * 3600, 300)], mode="power_only")
        result = analyse_decoupling(7 * 3600 - 300, power, hr, "endurance", 1.02)
        assert result.reason is None
        assert result.pct is not None
        assert result.window_s == 7 * 3600

    def test_many_short_stops_no_longer_add_up_to_a_refusal(self):
        # Eight six-minute stops: 11% of the ride recorded heart rate and no
        # watts, which cleared the old 5% budget several times over.
        stops = [(k * 3000 + 600, 360) for k in range(1, 9)]
        power, hr = _long_ride(stops=stops, mode="power_only")
        result = analyse_decoupling(7 * 3600, power, hr, "endurance", 1.02)
        assert result.reason is None
        assert result.pct is not None

    def test_a_device_pause_is_measured_across_as_before(self):
        power, hr = _long_ride(stops=[(3 * 3600, 300)], mode="shared")
        result = analyse_decoupling(7 * 3600 - 300, power, hr, "endurance", 1.02)
        assert result.reason is None
        assert result.window_s == 7 * 3600

    def test_a_lunch_stop_is_measured_over_the_longer_side(self):
        power, hr = _long_ride(stops=[(2 * 3600, 3600)])
        result = analyse_decoupling(6 * 3600, power, hr, "endurance", 1.02)
        assert result.reason is None
        assert result.window_s == 4 * 3600

    def test_a_dropping_strap_is_still_a_mismatch(self):
        # Heart rate missing while the meter reports watts is a strap problem,
        # and pairing what is left would compare two different rides.
        stops = [(k * 3000 + 600, 360) for k in range(1, 9)]
        power, hr = _long_ride(stops=stops, mode="hr_only")
        result = analyse_decoupling(7 * 3600, power, hr, "endurance", 1.02)
        assert result.reason == "stream_mismatch"
        assert result.pct is None
        assert result.window_s is None

    def test_a_ride_no_block_of_which_lasts_an_hour_is_fragmented(self):
        # Stop-start city riding: five hours long, in fifty-minute pieces.
        stops = [(k * 3600 + 3000, 900) for k in range(5)]
        power, hr = _long_ride(5 * 3600, stops=stops)
        result = analyse_decoupling(5 * 3600 - 5 * 900, power, hr, "endurance", 1.02)
        assert result.reason == "fragmented"
        assert result.pct is None

    def test_a_meter_that_dies_is_measured_over_what_it_recorded(self):
        """Deliberate: the block is the ride's longest, not a share of it.

        A meter whose battery went at 70 minutes used to report `no_power`,
        because the second half of the grid had none. Seventy minutes is an
        honest window and the figure now covers it — which is only safe because
        `window_s` travels with the number and says so.
        """
        power = [200.0 + (i % 7) for i in range(4200)] + [None] * (7 * 3600 - 4200)
        hr = [140.0 + (i % 5) * 0.1 + 4 * i / 25200 for i in range(7 * 3600)]
        result = analyse_decoupling(7 * 3600, power, hr, "endurance", 1.02)
        assert result.pct is not None
        assert result.window_s == 4200

    def test_a_genuinely_short_ride_is_still_short_not_fragmented(self):
        power, hr = _long_ride(1800)
        result = analyse_decoupling(1800, power, hr, "endurance", 1.02)
        assert result.reason == "too_short"

    def test_a_sparse_recording_is_short_rather_than_fragmented(self):
        # Four hours elapsed, forty minutes of readings, spread evenly: nothing
        # was left outside the block, so the recording is thin, not broken up.
        power = ([200.0] + [None] * 5) * 2400
        hr = ([150.0] + [None] * 5) * 2400
        assert analyse_decoupling(14400, power, hr, "endurance", 1.02).reason == "too_short"

    def test_the_gates_still_apply_inside_the_window(self):
        # A negative split *within* the surviving block is still refused: the
        # window narrows where the drift is measured, it does not lower the bar.
        power = [150.0] * 2000 + [None] * 1800 + [150.0] * 2000 + [200.0] * 2000
        hr = (
            [130.0 + (i % 3) for i in range(2000)]
            + [None] * 1800
            + [130.0 + (i % 3) for i in range(2000)]
            + [150.0 + (i % 3) for i in range(2000)]
        )
        result = analyse_decoupling(6000, power, hr, "endurance", 1.03)
        assert result.reason == "uneven_pacing"

    def test_a_figure_and_a_reason_are_never_both_set(self):
        for streams_in in (
            _long_ride(stops=[(3 * 3600, 300)], mode="power_only"),
            _long_ride(stops=[(2 * 3600, 3600)]),
            _long_ride(1800),
            ([], []),
            ([200.0] * 4000, []),
        ):
            result = analyse_decoupling(7200, *streams_in, "endurance", 1.02)
            assert (result.pct is None) != (result.reason is None)
            # The window travels with the figure, never with a refusal.
            assert (result.window_s is None) == (result.pct is None)


class TestPairedHalvesSplit:
    """The halves are equal in recorded time, not in grid position."""

    def test_a_stop_before_halfway_does_not_shrink_the_first_half(self):
        # Twenty minutes parked at 1000 s. Splitting the grid down the middle
        # would leave the first half with 1200 s of riding against the second
        # half's 2000 s; splitting on paired seconds gives both 1600 s. Power
        # steps up exactly where the true recorded midpoint falls, so a grid
        # split reads the step as drift and a paired split reads it as zero.
        first = [200.0] * 1000 + [None] * 1200 + [200.0] * 600
        second = [220.0] * 1600
        power = first + second
        hr = (
            [140.0] * 1000
            + [None] * 1200
            + [140.0] * 600
            + [154.0] * 1600
        )
        # 220/154 == 200/140: the ratio is identical either side of the step.
        assert aerobic_decoupling(power, hr) == pytest.approx(0.0, abs=1e-9)
