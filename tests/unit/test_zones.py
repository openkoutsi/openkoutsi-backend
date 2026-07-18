import pytest
from openkoutsi.zones import Zones, time_in_zones


class TestZones:
    def test_init_stores_zones(self):
        z = Zones((0, 120), (121, 150), (151, 185))
        assert z.zones == [(0, 120), (121, 150), (151, 185)]

    def test_zone_name(self):
        z = Zones((0, 100), (101, 200))
        assert z.zoneName(0) == "Z1"
        assert z.zoneName(1) == "Z2"

    def test_get_zone_mid_range(self):
        z = Zones((0, 120), (121, 150), (151, 200))
        assert z.getZone(130) == 1  # Z2

    def test_get_zone_exact_lower_bound(self):
        z = Zones((0, 120), (121, 150))
        assert z.getZone(121) == 1

    def test_get_zone_exact_upper_bound(self):
        z = Zones((0, 120), (121, 150))
        assert z.getZone(120) == 0

    def test_get_zone_below_first_zone_clamps_to_z1(self):
        z = Zones((50, 120), (121, 150))
        assert z.getZone(10) == 0

    def test_get_zone_above_last_zone_clamps_to_last(self):
        z = Zones((0, 120), (121, 150))
        assert z.getZone(999) == 1

    def test_validate_raises_when_upper_not_greater_than_lower(self):
        with pytest.raises(ValueError, match="upper bound"):
            Zones((100, 100))

    def test_validate_raises_when_zones_overlap(self):
        with pytest.raises(ValueError, match="upper bound"):
            Zones((0, 130), (120, 200))

    def test_single_zone(self):
        z = Zones((0, 300))
        assert z.getZone(150) == 0
        assert z.getZone(0) == 0
        assert z.getZone(300) == 0


class TestTimeInZones:
    _ZONES = [
        {"name": "Z1", "low": 0, "high": 120},
        {"name": "Z2", "low": 121, "high": 150},
        {"name": "Z3", "low": 151, "high": 185},
    ]

    def test_counts_one_second_per_sample(self):
        samples = [100] * 30 + [135] * 30 + [160] * 40
        assert time_in_zones(samples, self._ZONES) == {"Z1": 30, "Z2": 30, "Z3": 40}

    def test_total_matches_sample_count(self):
        samples = [110, 130, 170, 90]
        assert sum(time_in_zones(samples, self._ZONES).values()) == len(samples)

    def test_out_of_range_samples_clamp_into_nearest_zone(self):
        # Below Z1 → Z1; above Z3 → Z3.
        assert time_in_zones([-5, 999], self._ZONES) == {"Z1": 1, "Z3": 1}

    def test_falls_back_to_positional_name(self):
        zones = [{"low": 0, "high": 150}, {"low": 151, "high": 300}]
        assert time_in_zones([100, 200], zones) == {"Z1": 1, "Z2": 1}

    def test_empty_stream_returns_empty(self):
        assert time_in_zones([], self._ZONES) == {}
