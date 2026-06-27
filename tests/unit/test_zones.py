import pytest
from openkoutsi.zones import Zones


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
