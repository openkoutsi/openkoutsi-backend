"""Unit tests for backend/app/services/zone_times.py."""
from backend.app.services.zone_times import compute_zone_times

_HR_ZONES = [
    {"name": "Z1", "low": 0, "high": 120},
    {"name": "Z2", "low": 120, "high": 140},
    {"name": "Z3", "low": 140, "high": 160},
    {"name": "Z4", "low": 160, "high": 175},
    {"name": "Z5", "low": 175, "high": 200},
]

_POWER_ZONES = [
    {"name": "Z1 Recovery", "low": 0, "high": 140},
    {"name": "Z2 Endurance", "low": 140, "high": 190},
    {"name": "Z3 Tempo", "low": 190, "high": 220},
    {"name": "Z4 Threshold", "low": 220, "high": 240},
    {"name": "Z5 VO2max", "low": 240, "high": 270},
    {"name": "Z6 Anaerobic", "low": 270, "high": 300},
    {"name": "Z7 Neuromuscular", "low": 300, "high": 9999},
]


class TestComputeZoneTimes:
    def test_both_streams_produce_both_keys(self):
        result = compute_zone_times(
            {"heartrate": [100] * 30 + [150] * 60, "power": [100] * 30 + [200] * 60},
            _HR_ZONES,
            _POWER_ZONES,
        )
        assert result == {
            "hr": {"Z1": 30, "Z3": 60},
            "power": {"Z1 Recovery": 30, "Z3 Tempo": 60},
        }

    def test_only_the_streams_that_exist(self):
        result = compute_zone_times({"power": [200] * 10}, _HR_ZONES, _POWER_ZONES)
        assert result == {"power": {"Z3 Tempo": 10}}

    def test_zones_without_a_stream_are_skipped(self):
        result = compute_zone_times({"heartrate": [150] * 10}, None, _POWER_ZONES)
        assert result is None

    def test_nothing_computable_returns_none(self):
        # None rather than an empty dict, so callers can leave the snapshot
        # unset and let it be backfilled later.
        assert compute_zone_times({}, _HR_ZONES, _POWER_ZONES) is None
        assert compute_zone_times({"power": [200] * 5}, None, None) is None
        assert compute_zone_times({"power": []}, None, _POWER_ZONES) is None

    def test_snapshot_only_carries_zones_the_ride_touched(self):
        # The partial-snapshot property that the three-band mapping has to
        # cope with (issue #38).
        result = compute_zone_times({"power": [100] * 60}, None, _POWER_ZONES)
        assert result == {"power": {"Z1 Recovery": 60}}


class TestGappyStreams:
    """A gap is time in no zone, not time in Z1 (issue #76).

    Streams span the whole elapsed ride with ``None`` where a sensor recorded
    nothing. That value must be dropped before the integer cast inside
    ``time_in_zones``: NaN casts to INT64_MIN, which clamps into the lowest zone
    and would book a ten-minute strap dropout as ten minutes of recovery riding.
    """

    def test_gaps_are_not_counted_as_a_zone(self):
        result = compute_zone_times(
            {"heartrate": [150] * 60 + [None] * 30}, _HR_ZONES, None
        )
        assert result == {"hr": {"Z3": 60}}

    def test_a_gap_does_not_become_the_lowest_zone(self):
        result = compute_zone_times(
            {"heartrate": [180] * 10 + [None] * 100}, _HR_ZONES, None
        )
        assert result == {"hr": {"Z5": 10}}

    def test_totals_reflect_recorded_time_not_elapsed(self):
        result = compute_zone_times(
            {"power": [200] * 30 + [None] * 30 + [200] * 30}, None, _POWER_ZONES
        )
        assert sum(result["power"].values()) == 60

    def test_an_all_gap_stream_contributes_nothing(self):
        assert compute_zone_times({"heartrate": [None] * 50}, _HR_ZONES, None) is None
