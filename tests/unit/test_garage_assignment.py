"""Unit tests for bike automapping and the derived garage figures (issue #64).

The pure halves of :mod:`backend.app.services.garage`: normalising what a bike
claims, deciding which bike a ride belongs to, and turning a maintenance log
into component life. The database-backed halves are covered end to end in
``tests/integration/test_garage.py``; these pin the decisions themselves, where
each branch is one sentence rather than four HTTP calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

import pytest

from backend.app.services import garage
from openkoutsi.sport_matching import CYCLING_SPORT_TYPES


@dataclass
class FakeActivity:
    """Just the columns :func:`garage.assign` reads and writes."""

    sport_type: Optional[str] = "Ride"
    bike_id: Optional[str] = None
    bike_source: Optional[str] = None


@dataclass
class FakeBike:
    odometer_base_km: Optional[float] = None


@dataclass
class FakeEntry:
    id: str
    component: str
    performed_on: date
    odometer_km: Optional[float] = None
    created_at: datetime = field(
        default_factory=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)
    )


# ── What a bike may claim ──────────────────────────────────────────────────


class TestNormaliseDefaultSports:
    def test_spellings_fold_onto_one_claim(self):
        """`gravel_ride` and `GravelRide` must not become two claims of which
        only one ever fires."""
        assert garage.normalise_default_sports(
            ["gravel_ride", "GravelRide", "GRAVELRIDE"]
        ) == ["GravelRide"]

    def test_order_is_preserved(self):
        assert garage.normalise_default_sports(["EBikeRide", "Ride"]) == [
            "EBikeRide",
            "Ride",
        ]

    def test_every_cycling_sport_is_claimable(self):
        claimed = garage.normalise_default_sports(sorted(CYCLING_SPORT_TYPES))
        assert set(claimed) == set(CYCLING_SPORT_TYPES)

    def test_virtual_ride_is_claimable_like_any_other(self):
        """Settled deliberately: a trainer ride is done on a real bike, and an
        athlete whose trainer bike is a fourth bike can simply claim it there."""
        assert garage.normalise_default_sports(["VirtualRide"]) == ["VirtualRide"]

    @pytest.mark.parametrize("sport", ["Run", "Swim", "Hike", "WeightTraining"])
    def test_a_non_cycling_sport_is_rejected(self, sport):
        with pytest.raises(garage.SportClaimError):
            garage.normalise_default_sports([sport])

    def test_an_unknown_sport_is_rejected(self):
        with pytest.raises(garage.SportClaimError):
            garage.normalise_default_sports(["Unicycling"])

    def test_a_bare_string_is_rejected_rather_than_iterated_as_characters(self):
        with pytest.raises(garage.SportClaimError):
            garage.normalise_default_sports("Ride")

    def test_none_and_empty_both_mean_claims_nothing(self):
        """One spelling of "nothing", so the API's answers stay stable."""
        assert garage.normalise_default_sports(None) is None
        assert garage.normalise_default_sports([]) is None


# ── Which bike a ride belongs to ───────────────────────────────────────────


class TestAssign:
    CLAIMS = {"Ride": "road", "GravelRide": "gravel"}

    def test_a_claimed_sport_is_assigned(self):
        activity = FakeActivity(sport_type="Ride")
        assert garage.assign(activity, self.CLAIMS) == "road"
        assert activity.bike_id == "road"
        assert activity.bike_source == garage.SOURCE_AUTO

    def test_the_sport_is_normalised_before_it_is_matched(self):
        """A file written by another tool spells the sport its own way."""
        activity = FakeActivity(sport_type="gravel_ride")
        assert garage.assign(activity, self.CLAIMS) == "gravel"

    def test_an_unclaimed_sport_leaves_it_null_rather_than_guessing(self):
        activity = FakeActivity(sport_type="MountainBikeRide")
        assert garage.assign(activity, self.CLAIMS) is None
        assert activity.bike_id is None
        assert activity.bike_source is None

    @pytest.mark.parametrize("sport", ["Run", "Swim", "Yoga", None, "", "Nonsense"])
    def test_a_non_cycling_activity_is_never_assigned(self, sport):
        activity = FakeActivity(sport_type=sport)
        assert garage.assign(activity, {**self.CLAIMS, "Run": "road"}) is None
        assert activity.bike_id is None

    def test_a_manual_choice_is_never_overwritten(self):
        """The invariant the whole feature turns on."""
        activity = FakeActivity(
            sport_type="Ride", bike_id="gravel", bike_source=garage.SOURCE_MANUAL
        )
        assert garage.assign(activity, self.CLAIMS) is None
        assert activity.bike_id == "gravel"
        assert activity.bike_source == garage.SOURCE_MANUAL

    def test_a_manual_choice_survives_even_when_nothing_claims_the_sport(self):
        activity = FakeActivity(
            sport_type="MountainBikeRide",
            bike_id="gravel",
            bike_source=garage.SOURCE_MANUAL,
        )
        assert garage.assign(activity, self.CLAIMS) is None
        assert activity.bike_id == "gravel"

    def test_an_automatic_assignment_moves_when_the_claim_moves(self):
        activity = FakeActivity(
            sport_type="Ride", bike_id="old", bike_source=garage.SOURCE_AUTO
        )
        assert garage.assign(activity, self.CLAIMS) == "road"
        assert activity.bike_id == "road"

    def test_narrowing_a_claim_withdraws_an_automatic_assignment(self):
        activity = FakeActivity(
            sport_type="Ride", bike_id="road", bike_source=garage.SOURCE_AUTO
        )
        assert garage.assign(activity, {}) is None
        assert activity.bike_id is None
        assert activity.bike_source is None

    def test_a_retired_bike_keeps_the_rides_it_already_has(self):
        """The whole difference between retiring a bike and deleting it. A
        retired bike claims nothing, so without this a reprocess after
        retirement would quietly strip its history."""
        activity = FakeActivity(
            sport_type="Ride", bike_id="sold", bike_source=garage.SOURCE_AUTO
        )
        assert garage.assign(activity, {}, retired_ids={"sold"}) is None
        assert activity.bike_id == "sold"
        assert activity.bike_source == garage.SOURCE_AUTO

    def test_it_is_idempotent(self):
        activity = FakeActivity(sport_type="Ride")
        garage.assign(activity, self.CLAIMS)
        before = (activity.bike_id, activity.bike_source)
        garage.assign(activity, self.CLAIMS)
        assert (activity.bike_id, activity.bike_source) == before


# ── Distance ───────────────────────────────────────────────────────────────


class TestLifetime:
    def test_the_baseline_is_added_to_what_was_tracked(self):
        assert garage.lifetime_km(FakeBike(odometer_base_km=4200.0), 800.0) == 5000.0

    def test_no_baseline_is_zero_rather_than_unknown(self):
        """A bike with no baseline has a lifetime figure; it is just the tracked
        one. Treating it as unknown would leave the common case blank."""
        assert garage.lifetime_km(FakeBike(), 800.0) == 800.0

    def test_a_baseline_alone_is_a_complete_answer(self):
        assert garage.lifetime_km(FakeBike(odometer_base_km=4200.0), 0.0) == 4200.0


# ── Component life ─────────────────────────────────────────────────────────


class TestComponentSpans:
    def test_the_span_is_the_gap_between_consecutive_entries(self):
        first = FakeEntry("1", "tyres", date(2026, 1, 1), 1000.0)
        second = FakeEntry("2", "tyres", date(2026, 6, 1), 4200.0)
        spans = garage.component_spans([first, second], 6000.0)
        assert spans["1"]["previous_component_km"] is None
        assert spans["2"]["previous_component_km"] == 3200.0

    def test_the_current_entry_counts_forward_to_the_lifetime(self):
        """The open-ended case, and the number the athlete actually wants."""
        entry = FakeEntry("1", "tyres", date(2026, 6, 1), 4200.0)
        spans = garage.component_spans([entry], 6000.0)
        assert spans["1"]["km_since"] == 1800.0
        assert spans["1"]["is_current"] is True

    def test_components_do_not_close_each_others_spans(self):
        tyres = FakeEntry("1", "tyres", date(2026, 1, 1), 1000.0)
        chain = FakeEntry("2", "chain", date(2026, 3, 1), 2000.0)
        spans = garage.component_spans([tyres, chain], 3000.0)
        assert spans["2"]["previous_component_km"] is None
        assert spans["1"]["is_current"] is True
        assert spans["2"]["is_current"] is True

    def test_a_missing_earlier_reading_makes_the_span_unknown_not_zero(self):
        first = FakeEntry("1", "tyres", date(2026, 1, 1), None)
        second = FakeEntry("2", "tyres", date(2026, 6, 1), 4200.0)
        spans = garage.component_spans([first, second], 6000.0)
        assert spans["2"]["previous_component_km"] is None

    def test_a_missing_current_reading_makes_the_span_unknown_in_both_directions(self):
        first = FakeEntry("1", "tyres", date(2026, 1, 1), 1000.0)
        second = FakeEntry("2", "tyres", date(2026, 6, 1), None)
        spans = garage.component_spans([first, second], 6000.0)
        assert spans["2"]["previous_component_km"] is None
        assert spans["2"]["km_since"] is None

    def test_only_the_newest_entry_of_a_component_is_current(self):
        entries = [
            FakeEntry("1", "tyres", date(2026, 1, 1), 1000.0),
            FakeEntry("2", "tyres", date(2026, 6, 1), 4200.0),
            FakeEntry("3", "tyres", date(2026, 3, 1), 2500.0),
        ]
        spans = garage.component_spans(entries, 6000.0)
        assert [spans[i]["is_current"] for i in ("1", "2", "3")] == [False, True, False]

    def test_entries_out_of_order_are_sorted_before_the_spans_are_taken(self):
        """The API can hand the log over in any order; the arithmetic is about
        what happened when, not about what was inserted when."""
        entries = [
            FakeEntry("2", "tyres", date(2026, 6, 1), 4200.0),
            FakeEntry("1", "tyres", date(2026, 1, 1), 1000.0),
        ]
        spans = garage.component_spans(entries, 6000.0)
        assert spans["2"]["previous_component_km"] == 3200.0

    def test_same_day_entries_fall_back_to_the_order_they_were_recorded_in(self):
        """`performed_on` is a date, so two things done on one day tie — and an
        unstable order would make the numbers move between reads."""
        first = FakeEntry(
            "1",
            "tyres",
            date(2026, 6, 1),
            1000.0,
            created_at=datetime(2026, 6, 1, 9, tzinfo=timezone.utc),
        )
        second = FakeEntry(
            "2",
            "tyres",
            date(2026, 6, 1),
            1050.0,
            created_at=datetime(2026, 6, 1, 17, tzinfo=timezone.utc),
        )
        spans = garage.component_spans([second, first], 2000.0)
        assert spans["2"]["previous_component_km"] == 50.0
        assert spans["2"]["is_current"] is True

    def test_an_unknown_lifetime_leaves_the_running_figure_unknown(self):
        entry = FakeEntry("1", "tyres", date(2026, 6, 1), 4200.0)
        spans = garage.component_spans([entry], None)
        assert spans["1"]["km_since"] is None

    def test_an_empty_log_is_an_empty_answer(self):
        assert garage.component_spans([], 1000.0) == {}
