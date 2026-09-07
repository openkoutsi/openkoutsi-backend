"""Unit tests for `scripts/inspect_decoupling.py`.

The script exists to answer "why did my ride not get a figure?" against a real
file, so the part that classifies holes is the part worth testing: it has to
tell a device pause from a sleeping power meter from a dropped strap, since
those three look identical in a coverage percentage and mean entirely
different things.
"""
import pytest

from scripts.inspect_decoupling import Profile, profile_streams


class TestProfileStreams:
    def test_a_clean_ride_has_no_holes(self):
        profile = profile_streams([200.0] * 3600, [140.0] * 3600)
        assert profile.holes == []
        assert profile.paired_s == profile.power_s == profile.hr_s == 3600
        assert profile.power_anchored == 1.0

    def test_a_device_pause_is_one_hole_in_both_channels(self):
        power = [200.0] * 1000 + [None] * 300 + [200.0] * 1000
        hr = [140.0] * 1000 + [None] * 300 + [140.0] * 1000
        (hole,) = profile_streams(power, hr).holes
        assert (hole.start_s, hole.length_s, hole.code) == (1000, 300, 3)

    def test_a_sleeping_meter_and_a_dropped_strap_are_told_apart(self):
        power = [200.0] * 500 + [None] * 200 + [200.0] * 800
        hr = [140.0] * 1200 + [None] * 100 + [140.0] * 200
        holes = profile_streams(power, hr).holes
        assert [(h.start_s, h.length_s, h.code) for h in holes] == [
            (500, 200, 1),  # power only — the meter slept
            (1200, 100, 2),  # heart rate only — the strap dropped
        ]

    def test_holes_come_back_longest_first(self):
        power = [200.0] * 100 + [None] * 10 + [200.0] * 100 + [None] * 50 + [200.0] * 100
        hr = [140.0] * 360
        assert [h.length_s for h in profile_streams(power, hr).holes] == [50, 10]

    def test_the_two_denominators_disagree_exactly_where_the_bug_was(self):
        # Ten minutes of a one-hour ride with a pulse and no watts: the rider
        # standing still. Dividing by heart rate calls that a mismatch; dividing
        # by what the meter recorded calls it a stop.
        power = [200.0] * 3000 + [None] * 600
        hr = [140.0] * 3600
        profile = profile_streams(power, hr)
        assert profile.hr_anchored == pytest.approx(3000 / 3600)
        assert profile.power_anchored == 1.0

    def test_empty_streams_profile_to_nothing(self):
        assert profile_streams([], []) == Profile(0, 0, 0, 0, [])
        assert profile_streams([], [140.0] * 100).grid_s == 0
