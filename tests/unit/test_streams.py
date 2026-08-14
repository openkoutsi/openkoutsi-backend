"""The stream contract: every channel on one clock, gaps marked (issue #76).

The bug these guard against is quiet by construction. The old parser appended
one sample per record *that carried the field*, so a heart-rate dropout of n
records did not leave a hole — it shifted every later HR sample n positions
earlier relative to power. The lists stayed dense and merely ended up different
lengths, which no assertion in the suite was looking at.

So the tests here are mostly about *position*: not "is the data still there" but
"is each sample still on the second it was recorded at, in every channel".
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from openkoutsi import streams


_T0 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)


def _times(*offsets_s: float) -> list[datetime]:
    return [_T0 + timedelta(seconds=s) for s in offsets_s]


class TestSecondOffsets:
    def test_offsets_are_relative_to_the_first_record(self):
        assert streams.second_offsets(_times(0, 1, 2, 3)) == ([0, 1, 2, 3], 4)

    def test_a_hole_in_the_records_widens_the_grid(self):
        # Three records spanning ten seconds is a ten-second grid, not a
        # three-slot list. The seven missing seconds are the whole point.
        assert streams.second_offsets(_times(0, 5, 10)) == ([0, 5, 10], 11)

    def test_sub_second_records_collapse_onto_the_same_slot(self):
        assert streams.second_offsets(_times(0, 0.5, 1, 1.5)) == ([0, 0, 1, 1], 2)

    def test_empty_input(self):
        assert streams.second_offsets([]) == ([], 0)

    def test_records_beyond_the_cap_are_rejected_not_allocated(self):
        # A device that stamps one record a year out must not size a list from
        # it; the offset comes back as -1 for the caller to drop.
        offsets, length = streams.second_offsets(_times(0, 1, 400_000))
        assert offsets == [0, 1, -1]
        assert length == 2

    def test_a_clock_stepping_backwards_is_rejected(self):
        offsets, length = streams.second_offsets(_times(0, 5, -30, 6))
        assert offsets == [0, 5, -1, 6]
        assert length == 7


class TestResample1Hz:
    def test_dense_records_are_unchanged(self):
        channels = {"power": [(i, 200.0 + i) for i in range(4)]}
        assert streams.resample_1hz(channels, 4)["power"] == [200.0, 201.0, 202.0, 203.0]

    def test_a_dropout_becomes_a_hole_rather_than_a_shift(self):
        """The regression this whole change exists for.

        Heart rate is missing from records 2 and 3. Under the old parser the HR
        list was [140, 141, 144, 145] — dense, and 144 sat at index 2 where the
        power from second 2 was, pairing a wattage against a heartbeat recorded
        two seconds later. Now it stays at index 4.
        """
        channels = {
            "power": [(i, 200.0 + i) for i in range(6)],
            "heartrate": [(0, 140.0), (1, 141.0), (4, 144.0), (5, 145.0)],
        }
        out = streams.resample_1hz(channels, 6)
        assert out["heartrate"] == [140.0, 141.0, None, None, 144.0, 145.0]
        assert len(out["heartrate"]) == len(out["power"])

    def test_two_channels_dropping_at_different_points_stay_aligned(self):
        """The case the #74 length guard could not see.

        Power drops two records early in the ride, heart rate drops two late.
        The old parser produced two dense lists of equal length — the guard
        compared lengths and found nothing wrong — while every sample between
        the two dropouts was paired against the wrong second.
        """
        power = [(i, float(i)) for i in range(10) if i not in (2, 3)]
        hr = [(i, float(i)) for i in range(10) if i not in (7, 8)]
        out = streams.resample_1hz({"power": power, "heartrate": hr}, 10)

        assert len(out["power"]) == len(out["heartrate"]) == 10
        # Every second that has both channels has them from the *same* second.
        for i in range(10):
            p, h = out["power"][i], out["heartrate"][i]
            if p is not None and h is not None:
                assert p == h == float(i)
        assert out["power"][2] is None and out["heartrate"][2] == 2.0
        assert out["heartrate"][7] is None and out["power"][7] == 7.0

    def test_last_record_in_a_second_wins(self):
        out = streams.resample_1hz({"power": [(0, 100.0), (0, 150.0)]}, 1)
        assert out["power"] == [150.0]

    def test_a_channel_with_no_samples_is_empty_not_all_gaps(self):
        # "This activity has no power meter" and "this power meter recorded
        # nothing" are different claims, and callers key off the empty list.
        out = streams.resample_1hz({"power": [], "heartrate": [(0, 140.0)]}, 1)
        assert out["power"] == []
        assert out["heartrate"] == [140.0]

    def test_samples_past_the_grid_are_dropped(self):
        out = streams.resample_1hz({"power": [(0, 1.0), (99, 2.0)]}, 2)
        assert out["power"] == [1.0, None]


class TestResampleFromTimeStream:
    def test_irregular_provider_samples_land_on_their_seconds(self):
        out = streams.resample_from_time_stream(
            [0, 1, 5, 6], {"power": [100.0, 110.0, 150.0, 160.0]}
        )
        assert out["power"] == [100.0, 110.0, None, None, None, 150.0, 160.0]

    def test_channels_stay_aligned_with_each_other(self):
        out = streams.resample_from_time_stream(
            [0, 4], {"power": [100.0, 140.0], "heartrate": [150.0, 160.0]}
        )
        assert out["power"] == [100.0, None, None, None, 140.0]
        assert out["heartrate"] == [150.0, None, None, None, 160.0]

    def test_a_time_stream_not_starting_at_zero_is_still_zero_based(self):
        out = streams.resample_from_time_stream([10, 11], {"power": [1.0, 2.0]})
        assert out["power"] == [1.0, 2.0]

    def test_a_short_channel_contributes_only_what_it_has(self):
        out = streams.resample_from_time_stream(
            [0, 1, 2], {"power": [1.0, 2.0, 3.0], "heartrate": [140.0]}
        )
        assert out["heartrate"] == [140.0, None, None]

    def test_without_a_time_stream_the_arrays_pass_through(self):
        out = streams.resample_from_time_stream([], {"power": [1.0, 2.0]})
        assert out["power"] == [1.0, 2.0]


class TestGapReadings:
    """`present` and `filled` are the two legitimate ways to read a gap."""

    def test_present_drops_gaps(self):
        assert list(streams.present([1.0, None, 3.0])) == [1.0, 3.0]

    def test_filled_keeps_the_clock(self):
        assert list(streams.filled([1.0, None, 3.0])) == [1.0, 0.0, 3.0]

    def test_as_array_turns_gaps_into_nan(self):
        arr = streams.as_array([1.0, None])
        assert arr[0] == 1.0 and np.isnan(arr[1])

    def test_present_ratio(self):
        assert streams.present_ratio([1.0, None, 3.0, None]) == 0.5
        assert streams.present_ratio([]) == 0.0

    def test_paired_count_is_seconds_both_channels_cover(self):
        assert streams.paired_count([1.0, None, 3.0], [1.0, 2.0, None]) == 1
        assert streams.paired_count([1.0, 2.0], [1.0, 2.0]) == 2

    def test_paired_count_over_ragged_pre_issue_76_streams(self):
        # A stream stored before #76 has no gaps to count, it is just shorter.
        assert streams.paired_count([1.0] * 10, [1.0] * 4) == 4


class TestToJsonStream:
    def test_nan_becomes_null(self):
        assert streams.to_json_stream([1.0, float("nan"), 3.0]) == [1.0, None, 3.0]

    def test_none_stays_null(self):
        assert streams.to_json_stream([1.0, None]) == [1.0, None]

    def test_infinities_become_null_too(self):
        # json.dumps writes `Infinity`, which is as invalid as `NaN`.
        assert streams.to_json_stream([float("inf"), float("-inf")]) == [None, None]

    def test_the_result_survives_json_dumps(self):
        import json

        assert json.loads(json.dumps(streams.to_json_stream([1.0, float("nan")]))) == [
            1.0,
            None,
        ]

    @pytest.mark.parametrize("value", [None, []])
    def test_empty_inputs(self, value):
        assert streams.to_json_stream(value) == []
