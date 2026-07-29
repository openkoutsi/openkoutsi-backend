"""Unit tests for openkoutsi/intensity_distribution.py."""
import pytest

from openkoutsi.categorization import WorkoutCategory
from openkoutsi.intensity_distribution import (
    BAND_HIGH,
    BAND_LOW,
    BAND_MODERATE,
    PREDOMINANTLY_LOW,
    POLARIZED,
    PYRAMIDAL,
    THRESHOLD,
    _CATEGORY_BANDS,
    band_for_category,
    band_for_zone_index,
    band_percentages,
    bands_from_zone_times,
    classify,
)
from openkoutsi.zones import HR_ZONE_COUNT, POWER_ZONE_COUNT


class TestBandForZoneIndex:
    def test_canonical_power_model(self):
        bands = [band_for_zone_index(i, POWER_ZONE_COUNT, "power") for i in range(7)]
        assert bands == [
            BAND_LOW, BAND_LOW,
            BAND_MODERATE, BAND_MODERATE,
            BAND_HIGH, BAND_HIGH, BAND_HIGH,
        ]

    def test_canonical_hr_model(self):
        bands = [band_for_zone_index(i, HR_ZONE_COUNT, "hr") for i in range(5)]
        assert bands == [
            BAND_LOW, BAND_LOW,
            BAND_MODERATE, BAND_MODERATE,
            BAND_HIGH,
        ]

    def test_legacy_three_zone_list_splits_proportionally(self):
        # Snapshots frozen before the zone count was pinned down must still map
        # to something sensible rather than being dropped.
        bands = [band_for_zone_index(i, 3, "power") for i in range(3)]
        assert bands == [BAND_LOW, BAND_MODERATE, BAND_HIGH]

    def test_legacy_six_zone_list(self):
        # Boundaries land at the same proportions of the list as in the
        # canonical model — zone midpoints either side of 2/7 and 4/7.
        bands = [band_for_zone_index(i, 6, "power") for i in range(6)]
        assert bands == [
            BAND_LOW, BAND_LOW,
            BAND_MODERATE,
            BAND_HIGH, BAND_HIGH, BAND_HIGH,
        ]

    def test_single_zone_list_is_all_low(self):
        assert band_for_zone_index(0, 1, "power") == BAND_LOW

    def test_every_index_lands_in_a_band(self):
        for basis, canonical in (("power", POWER_ZONE_COUNT), ("hr", HR_ZONE_COUNT)):
            for count in range(1, 12):
                bands = {band_for_zone_index(i, count, basis) for i in range(count)}
                assert bands <= {BAND_LOW, BAND_MODERATE, BAND_HIGH}

    def test_unknown_basis_raises(self):
        with pytest.raises(ValueError):
            band_for_zone_index(0, 7, "pace")


class TestBandsFromZoneTimes:
    _POWER_SNAPSHOT = {
        "power": {
            "Z1 Recovery": 100,
            "Z2 Endurance": 200,
            "Z3 Tempo": 30,
            "Z4 Threshold": 40,
            "Z5 VO2max": 10,
            "Z6 Anaerobic": 5,
            "Z7 Neuromuscular": 1,
        }
    }

    def test_sums_into_three_bands(self):
        totals = bands_from_zone_times(self._POWER_SNAPSHOT, "power")
        assert totals == {BAND_LOW: 300, BAND_MODERATE: 70, BAND_HIGH: 16}

    def test_bare_zone_names_from_provider_sync(self):
        snapshot = {"power": {f"Z{i}": 10 for i in range(1, 8)}}
        assert bands_from_zone_times(snapshot, "power") == {
            BAND_LOW: 20, BAND_MODERATE: 20, BAND_HIGH: 30
        }

    def test_partial_snapshot_keeps_its_zone_numbers(self):
        # A ride that never left the easy zones stores only those keys. Reading
        # them as a three-zone model would call a recovery spin hard work.
        snapshot = {"power": {"Z1 Recovery": 600, "Z2 Endurance": 1200, "Z3 Tempo": 60}}
        assert bands_from_zone_times(snapshot, "power") == {
            BAND_LOW: 1800, BAND_MODERATE: 60, BAND_HIGH: 0
        }

    def test_single_top_zone_snapshot(self):
        assert bands_from_zone_times({"power": {"Z7": 45}}, "power") == {
            BAND_LOW: 0, BAND_MODERATE: 0, BAND_HIGH: 45
        }

    def test_ordering_is_numeric_not_lexicographic(self):
        # "Z10" must land in the top band, not beside "Z1".
        snapshot = {"power": {"Z1": 1, "Z2": 2, "Z3": 3, "Z4": 4, "Z5": 5, "Z6": 6, "Z10": 10}}
        totals = bands_from_zone_times(snapshot, "power")
        # Z10 proves a ten-zone list, so the bands widen proportionally: the
        # bottom three zones are easy and Z10's time is unambiguously hard.
        assert totals == {BAND_LOW: 6, BAND_MODERATE: 15, BAND_HIGH: 10}

    def test_other_basis_is_ignored(self):
        snapshot = {"power": {"Z1": 60}, "hr": {"Z5": 30}}
        assert bands_from_zone_times(snapshot, "hr") == {
            BAND_LOW: 0, BAND_MODERATE: 0, BAND_HIGH: 30
        }

    def test_missing_and_empty_snapshots(self):
        empty = {BAND_LOW: 0, BAND_MODERATE: 0, BAND_HIGH: 0}
        assert bands_from_zone_times(None, "power") == empty
        assert bands_from_zone_times({}, "power") == empty
        assert bands_from_zone_times({"hr": {}}, "hr") == empty


class TestBandForCategory:
    def test_every_category_is_mapped(self):
        # A new WorkoutCategory must be an explicit decision, not a silent
        # default into band 1.
        assert set(_CATEGORY_BANDS) == set(WorkoutCategory)

    @pytest.mark.parametrize(
        "category,expected",
        [
            (WorkoutCategory.recovery, BAND_LOW),
            (WorkoutCategory.endurance, BAND_LOW),
            (WorkoutCategory.tempo, BAND_MODERATE),
            (WorkoutCategory.threshold, BAND_MODERATE),
            (WorkoutCategory.vo2max, BAND_HIGH),
            (WorkoutCategory.anaerobic, BAND_HIGH),
            (WorkoutCategory.sprint, BAND_HIGH),
        ],
    )
    def test_cycling_categories(self, category, expected):
        assert band_for_category(category) == expected

    @pytest.mark.parametrize(
        "category",
        [WorkoutCategory.strength, WorkoutCategory.yoga, WorkoutCategory.cross_training],
    )
    def test_non_cycling_categories_excluded(self, category):
        assert band_for_category(category) is None

    def test_accepts_raw_strings(self):
        assert band_for_category("endurance") == BAND_LOW
        assert band_for_category("vo2max") == BAND_HIGH

    def test_unset_and_unknown_excluded(self):
        assert band_for_category(None) is None
        assert band_for_category("not_a_category") is None


class TestBandPercentages:
    def test_shares_of_the_total(self):
        pct = band_percentages({BAND_LOW: 60, BAND_MODERATE: 30, BAND_HIGH: 10})
        assert pct == {BAND_LOW: 60.0, BAND_MODERATE: 30.0, BAND_HIGH: 10.0}

    def test_all_zero_does_not_divide_by_zero(self):
        assert band_percentages({BAND_LOW: 0, BAND_MODERATE: 0, BAND_HIGH: 0}) == {
            BAND_LOW: 0.0, BAND_MODERATE: 0.0, BAND_HIGH: 0.0
        }


class TestClassify:
    def test_empty_window_has_no_shape(self):
        assert classify(0, 0, 0) is None

    def test_predominantly_low(self):
        assert classify(95, 3, 2) == PREDOMINANTLY_LOW

    def test_low_intensity_guard_boundary(self):
        # Exactly at the guard is no longer "almost nothing above LT1".
        assert classify(90, 5, 5) != PREDOMINANTLY_LOW
        assert classify(91, 5, 4) == PREDOMINANTLY_LOW

    def test_polarized(self):
        assert classify(80, 5, 15) == POLARIZED

    def test_pyramidal(self):
        assert classify(75, 18, 7) == PYRAMIDAL

    def test_threshold_by_share(self):
        assert classify(55, 35, 10) == THRESHOLD

    def test_threshold_when_band_two_is_largest(self):
        assert classify(32, 34, 34) == THRESHOLD

    def test_threshold_wins_over_polarized(self):
        # The grey-zone grind this feature exists to expose: lots of band 2,
        # marginally more band 3. Calling it polarized would bury the finding.
        assert classify(20, 38, 42) == THRESHOLD

    def test_degenerate_all_low(self):
        assert classify(100, 0, 0) == PREDOMINANTLY_LOW

    def test_degenerate_all_moderate(self):
        assert classify(0, 100, 0) == THRESHOLD

    def test_degenerate_all_high(self):
        # No label fits a block that is entirely above LT2. Pinned so the
        # behaviour cannot drift unnoticed.
        assert classify(0, 0, 100) == POLARIZED
