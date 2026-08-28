"""Unit tests for openkoutsi/commute.py (issue #63)."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from openkoutsi.commute import (
    CommuteRule,
    RideSample,
    TimeWindow,
    format_time_of_day,
    match_commute,
    near_miss_criteria,
    parse_rule,
    parse_rules,
    parse_time_of_day,
    propose_rule,
)

HELSINKI = ZoneInfo("Europe/Helsinki")


def _rule(**overrides) -> CommuteRule:
    """A plausible morning/evening bike commute, tweakable per test."""
    base = dict(
        id="commute",
        sport_types=frozenset({"ride"}),
        min_distance_m=4000.0,
        max_distance_m=8000.0,
        windows=(TimeWindow(6 * 60 + 30, 8 * 60 + 30), TimeWindow(15 * 60 + 30, 18 * 60)),
        weekdays=frozenset({0, 1, 2, 3, 4}),
    )
    base.update(overrides)
    return CommuteRule(**base)


def _match(rule: CommuteRule, **kwargs):
    defaults = dict(
        sport_type="Ride",
        start_time=datetime(2026, 8, 26, 7, 30),  # a Wednesday
        duration_s=1200,
        distance_m=5400.0,
    )
    defaults.update(kwargs)
    return match_commute([rule], **defaults)


class TestParseTimeOfDay:
    def test_parses_hh_mm(self):
        assert parse_time_of_day("07:30") == 450
        assert parse_time_of_day("00:00") == 0
        assert parse_time_of_day("23:59") == 1439

    def test_tolerates_whitespace_and_short_hours(self):
        assert parse_time_of_day(" 7:05 ") == 425

    def test_accepts_an_already_converted_minute_count(self):
        assert parse_time_of_day(450) == 450

    @pytest.mark.parametrize(
        "raw", ["24:00", "07:60", "25:10", "seven", "", "7", "07:5", None, True, 1440, -1, 3.5]
    )
    def test_rejects_nonsense(self, raw):
        assert parse_time_of_day(raw) is None

    def test_round_trips_through_format(self):
        for minute in (0, 1, 450, 1439):
            assert parse_time_of_day(format_time_of_day(minute)) == minute


class TestTimeWindow:
    def test_inclusive_of_both_edges(self):
        w = TimeWindow(400, 500)
        assert w.contains(400) and w.contains(500) and w.contains(450)
        assert not w.contains(399) and not w.contains(501)

    def test_spanning_midnight_is_not_an_inverted_range(self):
        """22:00-02:00 is a night-shift commute, not a mistake."""
        w = TimeWindow(22 * 60, 2 * 60)
        assert w.contains(23 * 60)
        assert w.contains(0)
        assert w.contains(90)
        assert not w.contains(12 * 60)


class TestMatching:
    def test_matches_a_ride_satisfying_every_criterion(self):
        assert _match(_rule()) is not None

    def test_sport_type_is_matched_exactly_but_case_insensitively(self):
        assert _match(_rule(), sport_type="ride") is not None
        assert _match(_rule(), sport_type="RIDE") is not None
        assert _match(_rule(), sport_type="VirtualRide") is None

    def test_ebike_can_be_the_whole_rule(self):
        """Issue #63: exact sport type is a supported criterion on its own."""
        ebike = CommuteRule(id="ebike", sport_types=frozenset({"ebikeride"}))
        assert _match(ebike, sport_type="EBikeRide") is not None
        assert _match(ebike, sport_type="Ride") is None

    @pytest.mark.parametrize("distance", [3999.0, 8001.0])
    def test_distance_outside_the_band_does_not_match(self, distance):
        assert _match(_rule(), distance_m=distance) is None

    @pytest.mark.parametrize("distance", [4000.0, 8000.0])
    def test_distance_band_is_inclusive(self, distance):
        assert _match(_rule(), distance_m=distance) is not None

    def test_duration_outside_the_band_does_not_match(self):
        rule = _rule(min_duration_s=600, max_duration_s=1500)
        assert _match(rule, duration_s=1499) is not None
        assert _match(rule, duration_s=1501) is None

    def test_outside_every_window_does_not_match(self):
        assert _match(_rule(), start_time=datetime(2026, 8, 26, 12, 0)) is None

    def test_the_evening_leg_matches_its_own_window(self):
        assert _match(_rule(), start_time=datetime(2026, 8, 26, 16, 45)) is not None

    def test_weekend_does_not_match_a_weekday_rule(self):
        assert _match(_rule(), start_time=datetime(2026, 8, 29, 7, 30)) is None  # Saturday

    def test_failing_exactly_one_criterion_is_enough_to_miss(self):
        """Criteria are AND, so each one alone can veto the match."""
        assert _match(_rule(), sport_type="Run") is None
        assert _match(_rule(), distance_m=20000.0) is None
        assert _match(_rule(), start_time=datetime(2026, 8, 26, 3, 0)) is None

    def test_missing_data_fails_the_criterion_it_cannot_satisfy(self):
        """A ride with no distance is not *shown* to be in the band."""
        assert _match(_rule(), distance_m=None) is None
        assert _match(_rule(), start_time=None) is None
        assert _match(_rule(), sport_type=None) is None

    def test_missing_data_is_irrelevant_to_criteria_that_are_not_set(self):
        rule = CommuteRule(id="anytime", sport_types=frozenset({"ride"}))
        assert _match(rule, distance_m=None, duration_s=None) is not None

    def test_disabled_rules_never_match(self):
        assert _match(_rule(enabled=False)) is None

    def test_first_matching_rule_wins(self):
        first = CommuteRule(id="first", sport_types=frozenset({"ride"}))
        second = CommuteRule(id="second", sport_types=frozenset({"ride"}))
        got = match_commute(
            [first, second],
            sport_type="Ride",
            start_time=datetime(2026, 8, 26, 7, 30),
            duration_s=1200,
            distance_m=5400.0,
        )
        assert got is not None and got.id == "first"

    def test_no_rules_means_no_match(self):
        assert (
            match_commute(
                [],
                sport_type="Ride",
                start_time=datetime(2026, 8, 26, 7, 30),
                duration_s=1200,
                distance_m=5400.0,
            )
            is None
        )


class TestEmptyRuleMatchesNothing:
    """The generous reading of "no criteria" would label a whole history."""

    def test_a_rule_with_no_criteria_has_no_criteria(self):
        assert CommuteRule(id="empty").has_criteria is False

    def test_and_therefore_matches_nothing(self):
        assert _match(CommuteRule(id="empty")) is None

    def test_and_is_dropped_at_parse_time(self):
        assert parse_rule({"id": "empty"}) is None
        assert parse_rules([{"id": "empty"}]) == []


class TestTimezoneHandling:
    """Windows are the athlete's local clock; start_time is UTC."""

    def test_utc_start_is_converted_before_the_window_is_checked(self):
        # 05:30 UTC is 08:30 in Helsinki (summer, UTC+3) — inside the window.
        start = datetime(2026, 8, 26, 5, 30, tzinfo=timezone.utc)
        assert _match(_rule(), start_time=start, tz=HELSINKI) is not None

    def test_without_a_zone_the_same_ride_falls_outside(self):
        """Proof the conversion is what is doing the work, not luck."""
        start = datetime(2026, 8, 26, 5, 30, tzinfo=timezone.utc)
        assert _match(_rule(), start_time=start, tz=None) is None

    def test_dst_boundary_keeps_a_window_meaning_the_same_local_time(self):
        """Helsinki is UTC+3 in August and UTC+2 in December.

        The same 07:45 local departure has different UTC stamps either side of
        the change; both must match, or commutes go silently undetected for half
        the year.
        """
        summer = datetime(2026, 8, 26, 4, 45, tzinfo=timezone.utc)   # 07:45 EEST
        winter = datetime(2026, 12, 16, 5, 45, tzinfo=timezone.utc)  # 07:45 EET
        assert _match(_rule(), start_time=summer, tz=HELSINKI) is not None
        assert _match(_rule(), start_time=winter, tz=HELSINKI) is not None

    def test_conversion_can_move_the_weekday_too(self):
        """23:30 UTC Sunday is Monday 02:30 in Helsinki."""
        rule = CommuteRule(id="night", weekdays=frozenset({0}), windows=(TimeWindow(120, 180),))
        start = datetime(2026, 8, 30, 23, 30, tzinfo=timezone.utc)  # Sunday in UTC
        assert _match(rule, start_time=start, tz=HELSINKI) is not None

    def test_a_naive_start_is_left_alone(self):
        """Converting a naive stamp would be inventing an offset."""
        assert _match(_rule(), start_time=datetime(2026, 8, 26, 7, 30), tz=HELSINKI) is not None

    def test_midnight_spanning_window_matches_either_side(self):
        rule = CommuteRule(id="nightshift", windows=(TimeWindow(22 * 60, 2 * 60),))
        assert _match(rule, start_time=datetime(2026, 8, 26, 23, 10)) is not None
        assert _match(rule, start_time=datetime(2026, 8, 26, 1, 10)) is not None
        assert _match(rule, start_time=datetime(2026, 8, 26, 12, 10)) is None


class TestParsing:
    def test_reads_a_full_rule(self):
        rule = parse_rule(
            {
                "id": "morning",
                "name": "To work",
                "sport_types": ["Ride", "EBikeRide"],
                "min_distance_m": 4000,
                "max_distance_m": 8000,
                "min_duration_s": 600,
                "max_duration_s": 1800,
                "windows": [{"start": "06:30", "end": "08:30"}],
                "weekdays": [0, 1, 2, 3, 4],
                "auto_apply": True,
            }
        )
        assert rule is not None
        assert rule.id == "morning"
        assert rule.name == "To work"
        assert rule.sport_types == frozenset({"ride", "ebikeride"})
        assert rule.windows == (TimeWindow(390, 510),)
        assert rule.weekdays == frozenset({0, 1, 2, 3, 4})
        assert rule.auto_apply is True
        assert rule.enabled is True

    def test_round_trips_through_as_dict(self):
        original = _rule(name="To work", auto_apply=True)
        assert parse_rule(original.as_dict()) == original

    def test_auto_apply_defaults_off(self):
        """A guess does not write to athlete-owned data unasked."""
        rule = parse_rule({"id": "x", "sport_types": ["Ride"]})
        assert rule is not None and rule.auto_apply is False

    def test_enabled_defaults_on(self):
        rule = parse_rule({"id": "x", "sport_types": ["Ride"]})
        assert rule is not None and rule.enabled is True

    def test_an_unreadable_field_costs_only_that_field(self):
        rule = parse_rule(
            {"id": "x", "sport_types": ["Ride"], "max_distance_m": "eight thousand"}
        )
        assert rule is not None
        assert rule.sport_types == frozenset({"ride"})
        assert rule.max_distance_m is None

    def test_an_unreadable_window_is_skipped_not_fatal(self):
        rule = parse_rule(
            {
                "id": "x",
                "windows": [{"start": "06:30", "end": "08:30"}, {"start": "nope", "end": "08:30"}],
            }
        )
        assert rule is not None and rule.windows == (TimeWindow(390, 510),)

    def test_inverted_ranges_are_dropped_rather_than_left_unsatisfiable(self):
        rule = parse_rule(
            {"id": "x", "sport_types": ["Ride"], "min_distance_m": 9000, "max_distance_m": 4000}
        )
        assert rule is not None
        assert rule.min_distance_m is None and rule.max_distance_m is None

    def test_negative_bounds_are_dropped(self):
        rule = parse_rule({"id": "x", "sport_types": ["Ride"], "min_distance_m": -5})
        assert rule is not None and rule.min_distance_m is None

    @pytest.mark.parametrize(
        "raw",
        [None, "a string", 42, {}, {"id": ""}, {"id": "   "}, {"id": 7, "sport_types": ["Ride"]}],
    )
    def test_unusable_rules_return_none(self, raw):
        assert parse_rule(raw) is None

    def test_weekday_values_out_of_range_are_dropped(self):
        rule = parse_rule({"id": "x", "sport_types": ["Ride"], "weekdays": [0, 7, -1, "mon", True]})
        assert rule is not None and rule.weekdays == frozenset({0})


class TestParseRulesIsTotal:
    """app_settings is unvalidated; a typo must not break the ingest path."""

    @pytest.mark.parametrize(
        "raw", [None, {}, "commute", 7, True, [None], ["nonsense"], [{"no": "id"}], [[]]]
    )
    def test_never_raises_and_degrades_to_no_rules(self, raw):
        assert parse_rules(raw) == []

    def test_keeps_the_good_rules_alongside_the_bad(self):
        rules = parse_rules([{"id": "ok", "sport_types": ["Ride"]}, "nonsense", {"id": "empty"}])
        assert [r.id for r in rules] == ["ok"]


class TestProposeRule:
    @staticmethod
    def _commute_history(count: int = 20) -> list[RideSample]:
        """A fortnight of weekday there-and-back rides, morning and evening."""
        samples = []
        day = datetime(2026, 8, 3, 0, 0)  # a Monday
        made = 0
        while made < count:
            if day.weekday() < 5:
                samples.append(
                    RideSample("Ride", day.replace(hour=7, minute=40), 1150, 5300.0)
                )
                made += 1
                if made < count:
                    samples.append(
                        RideSample("Ride", day.replace(hour=16, minute=50), 1320, 5800.0)
                    )
                    made += 1
            day += timedelta(days=1)
        return samples[:count]

    def test_needs_ten_rides_before_proposing_anything(self):
        """Issue #63: below this a cluster is a coincidence."""
        assert propose_rule(self._commute_history(9)) is None
        assert propose_rule(self._commute_history(10)) is not None

    def test_recovers_two_windows_from_a_there_and_back_history(self):
        rule = propose_rule(self._commute_history(20))
        assert rule is not None
        assert len(rule.windows) == 2
        starts = sorted(w.start for w in rule.windows)
        assert 7 * 60 - 30 < starts[0] < 7 * 60 + 40
        assert 16 * 60 < starts[1] < 17 * 60

    def test_recovers_a_distance_band_that_covers_the_history(self):
        rule = propose_rule(self._commute_history(20))
        assert rule is not None
        assert rule.min_distance_m is not None and rule.max_distance_m is not None
        assert rule.min_distance_m <= 5300.0
        assert rule.max_distance_m >= 5800.0

    def test_pins_weekdays_only_when_the_history_avoids_part_of_the_week(self):
        weekdays_only = propose_rule(self._commute_history(20))
        assert weekdays_only is not None
        assert weekdays_only.weekdays == frozenset({0, 1, 2, 3, 4})

        every_day = [
            RideSample("Ride", datetime(2026, 8, 3, 7, 40) + timedelta(days=i), 1150, 5300.0)
            for i in range(14)
        ]
        rule = propose_rule(every_day)
        assert rule is not None and rule.weekdays == frozenset()

    def test_a_proposed_rule_matches_the_rides_it_came_from(self):
        """The point of the proposal: it should work out of the box."""
        history = self._commute_history(20)
        rule = propose_rule(history)
        assert rule is not None
        for sample in history:
            assert rule.matches(
                sport_type=sample.sport_type,
                local_start=sample.local_start,
                duration_s=sample.duration_s,
                distance_m=sample.distance_m,
            ), sample

    def test_a_long_weekend_ride_does_not_match_the_proposal(self):
        rule = propose_rule(self._commute_history(20))
        assert rule is not None
        assert not rule.matches(
            sport_type="Ride",
            local_start=datetime(2026, 8, 8, 10, 0),  # Saturday
            duration_s=12000,
            distance_m=90000.0,
        )

    def test_one_long_detour_does_not_stretch_the_band_around_itself(self):
        history = self._commute_history(20)
        history[0] = RideSample("Ride", datetime(2026, 8, 3, 7, 40), 4000, 30000.0)
        rule = propose_rule(history)
        assert rule is not None and rule.max_distance_m is not None
        assert rule.max_distance_m < 15000.0

    def test_a_night_shift_history_yields_one_window_not_two(self):
        """Clock arithmetic is circular: 23:40 and 00:20 are 40 minutes apart."""
        samples = []
        for i in range(12):
            day = datetime(2026, 8, 3) + timedelta(days=i)
            samples.append(RideSample("Ride", day.replace(hour=23, minute=40), 1100, 5200.0))
            samples.append(
                RideSample("Ride", (day + timedelta(days=1)).replace(hour=0, minute=20), 1100, 5200.0)
            )
        rule = propose_rule(samples)
        assert rule is not None
        assert len(rule.windows) == 1
        assert rule.windows[0].start > rule.windows[0].end  # it wraps midnight

    def test_proposes_nothing_from_an_empty_history(self):
        assert propose_rule([]) is None

    def test_pins_sport_type_when_the_history_is_consistent(self):
        rule = propose_rule(self._commute_history(20))
        assert rule is not None and rule.sport_types == frozenset({"ride"})

    def test_does_not_pin_sport_type_across_a_wide_mix(self):
        samples = [
            RideSample(sport, datetime(2026, 8, 3, 7, 40) + timedelta(days=i), 1150, 5300.0)
            for i, sport in enumerate(
                ["Ride", "EBikeRide", "GravelRide", "VirtualRide", "MountainBikeRide"] * 3
            )
        ]
        rule = propose_rule(samples)
        assert rule is not None and rule.sport_types == frozenset()

    def test_survives_samples_with_missing_fields(self):
        samples = [
            RideSample(None, datetime(2026, 8, 3, 7, 40) + timedelta(days=i), None, None)
            for i in range(12)
        ]
        rule = propose_rule(samples)
        assert rule is not None and rule.windows


class TestNearMissCriteria:
    """The signal behind "your rule is too narrow"."""

    def test_a_perfect_match_fails_nothing(self):
        assert (
            near_miss_criteria(
                _rule(),
                sport_type="Ride",
                local_start=datetime(2026, 8, 26, 7, 30),
                duration_s=1200,
                distance_m=5400.0,
            )
            == []
        )

    def test_names_the_single_bound_that_missed(self):
        failed = near_miss_criteria(
            _rule(),
            sport_type="Ride",
            local_start=datetime(2026, 8, 26, 7, 30),
            duration_s=1200,
            distance_m=8600.0,
        )
        assert failed == ["distance"]

    def test_a_genuinely_different_ride_fails_several(self):
        failed = near_miss_criteria(
            _rule(),
            sport_type="Run",
            local_start=datetime(2026, 8, 29, 10, 0),  # Saturday, midday
            duration_s=1200,
            distance_m=42000.0,
        )
        assert set(failed) >= {"sport_types", "distance", "weekdays", "windows"}

    def test_missing_data_counts_as_failing_the_criterion(self):
        failed = near_miss_criteria(
            _rule(), sport_type="Ride", local_start=None, duration_s=1200, distance_m=5400.0
        )
        assert set(failed) == {"weekdays", "windows"}
