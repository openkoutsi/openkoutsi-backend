"""Unit tests for the pure achievement catalogue and streak math (issue #33)."""

from datetime import date, timedelta

import pytest

from openkoutsi.achievements import (
    CATALOGUE,
    CATALOGUE_BY_ID,
    COMEBACK_GAP_DAYS,
    ActivityFact,
    PeriodBucket,
    bucket_by,
    comeback_date,
    cumulative_tier_dates,
    distinct_tier_dates,
    month_start,
    qualifies_active,
    qualifies_multisport,
    qualifies_volume,
    streak_state,
    streak_tier_dates,
    threshold_tier_dates,
    week_start,
)


# ── Catalogue sanity ─────────────────────────────────────────────────────────


class TestCatalogue:
    def test_ids_are_unique(self):
        ids = [d.id for d in CATALOGUE]
        assert len(ids) == len(set(ids))

    def test_tiers_are_ascending_and_non_empty(self):
        for d in CATALOGUE:
            assert d.tiers, f"{d.id} has no tiers"
            assert list(d.tiers) == sorted(d.tiers), f"{d.id} tiers not ascending"

    def test_no_human_readable_strings_leak_into_definitions(self):
        """Names/descriptions are i18n keys in the web repo, never stored here."""
        for d in CATALOGUE:
            assert not hasattr(d, "name")
            assert not hasattr(d, "description")
            # ids are stable machine keys: lowercase, snake_case
            assert d.id == d.id.lower().replace(" ", "_")

    def test_lookup_covers_every_definition(self):
        assert set(CATALOGUE_BY_ID) == {d.id for d in CATALOGUE}

    def test_no_daily_streaks_are_offered(self):
        """The issue rules out daily streaks; the catalogue must not sneak one in."""
        for d in CATALOGUE:
            if d.category == "streak":
                assert d.unit in ("weeks", "months")

    def test_every_streak_definition_has_a_rule_and_vice_versa(self):
        """The catalogue and the service's rule table must agree, both ways.

        Drift fails late and quietly otherwise: a catalogue entry with no rule is
        a badge that can never be earned, and a rule with no entry has no tiers
        to compare against.
        """
        from backend.app.services.achievements import _STREAK_RULES

        assert set(_STREAK_RULES) == {
            d.id for d in CATALOGUE if d.category == "streak"
        }

    def test_every_tier_is_exactly_representable_as_a_float(self):
        """Tiers are part of a composite primary key, matched on equality.

        A tier that doesn't round-trip through float (0.1, say) would make the
        reconcile miss its own stored row, deleting and re-inserting it on every
        recompute — and re-announcing the badge each time.
        """
        for d in CATALOGUE:
            for tier in d.tiers:
                assert float(tier) == tier
                # Survives the round-trip the DB will put it through.
                assert float(repr(float(tier))) == float(tier)


# ── Week bucketing ───────────────────────────────────────────────────────────


class TestBucketing:
    def test_week_start_is_monday(self):
        # 2026-07-22 is a Wednesday
        assert week_start(date(2026, 7, 22)) == date(2026, 7, 20)
        assert week_start(date(2026, 7, 20)) == date(2026, 7, 20)  # Monday itself
        assert week_start(date(2026, 7, 26)) == date(2026, 7, 20)  # Sunday

    def test_sunday_and_monday_fall_in_different_weeks(self):
        sunday = date(2026, 7, 26)
        monday = date(2026, 7, 27)
        assert week_start(sunday) != week_start(monday)

    def test_bucket_sums_and_collects_sports(self):
        facts = [
            ActivityFact(day=date(2026, 7, 20), duration_s=3600, distance_m=30_000, sport="cycling"),
            ActivityFact(day=date(2026, 7, 22), duration_s=1800, distance_m=5_000, sport="running"),
            ActivityFact(day=date(2026, 7, 27), duration_s=7200, distance_m=60_000, sport="cycling"),
        ]
        buckets = bucket_by(facts)
        assert len(buckets) == 2
        assert buckets[0].start == date(2026, 7, 20)
        assert buckets[0].seconds == 5400
        assert buckets[0].metres == 35_000
        assert buckets[0].sports == {"cycling", "running"}
        assert buckets[0].count == 2
        assert buckets[1].start == date(2026, 7, 27)

    def test_buckets_are_ordered(self):
        facts = [
            ActivityFact(day=date(2026, 7, 27)),
            ActivityFact(day=date(2026, 6, 1)),
            ActivityFact(day=date(2026, 7, 20)),
        ]
        starts = [b.start for b in bucket_by(facts)]
        assert starts == sorted(starts)

    def test_monthly_bucketing(self):
        facts = [
            ActivityFact(day=date(2026, 7, 3)),
            ActivityFact(day=date(2026, 7, 28)),
            ActivityFact(day=date(2026, 8, 2)),
        ]
        buckets = bucket_by(facts, monthly=True)
        assert [b.start for b in buckets] == [date(2026, 7, 1), date(2026, 8, 1)]
        assert buckets[0].count == 2

    def test_month_start(self):
        assert month_start(date(2026, 7, 22)) == date(2026, 7, 1)


# ── Generic tier helpers ─────────────────────────────────────────────────────


class TestTierHelpers:
    def test_cumulative_reaches_tiers_on_the_crossing_day(self):
        events = [
            (date(2026, 1, 1), 1),
            (date(2026, 1, 5), 1),
            (date(2026, 1, 9), 1),
        ]
        reached = cumulative_tier_dates(events, [1, 2, 3, 10])
        assert reached[1] == date(2026, 1, 1)
        assert reached[2] == date(2026, 1, 5)
        assert reached[3] == date(2026, 1, 9)
        assert 10 not in reached

    def test_cumulative_sorts_unordered_input(self):
        events = [(date(2026, 1, 9), 1), (date(2026, 1, 1), 1)]
        assert cumulative_tier_dates(events, [1])[1] == date(2026, 1, 1)

    def test_cumulative_is_empty_for_no_events(self):
        assert cumulative_tier_dates([], [1, 2]) == {}

    def test_threshold_needs_a_single_value_to_clear_the_bar(self):
        events = [(date(2026, 1, 1), 1.5), (date(2026, 1, 2), 1.5)]
        # Two 1.5h rides never make a 3h ride.
        assert threshold_tier_dates(events, [3]) == {}
        assert threshold_tier_dates(events, [1]) == {1: date(2026, 1, 1)}

    def test_threshold_records_the_first_qualifying_day(self):
        events = [
            (date(2026, 1, 1), 2.0),
            (date(2026, 2, 1), 6.0),
            (date(2026, 3, 1), 4.0),
        ]
        reached = threshold_tier_dates(events, [2, 4, 6])
        assert reached[2] == date(2026, 1, 1)
        assert reached[4] == date(2026, 2, 1)
        assert reached[6] == date(2026, 2, 1)

    def test_distinct_counts_unique_values_only(self):
        events = [
            (date(2026, 1, 1), "cycling"),
            (date(2026, 1, 2), "cycling"),
            (date(2026, 1, 3), "running"),
            (date(2026, 1, 4), None),
            (date(2026, 1, 5), "swimming"),
        ]
        reached = distinct_tier_dates(events, [2, 3])
        assert reached[2] == date(2026, 1, 3)
        assert reached[3] == date(2026, 1, 5)


# ── Streaks ──────────────────────────────────────────────────────────────────


def _weeks(*offsets: int, base: date = date(2026, 1, 5), **kwargs) -> list[PeriodBucket]:
    """Buckets for the given week offsets from *base* (a Monday)."""
    return [
        PeriodBucket(start=base + timedelta(weeks=o), count=1, **kwargs) for o in offsets
    ]


class TestStreakState:
    def test_empty_history_has_no_streak(self):
        state = streak_state([], qualifies_active, date(2026, 2, 2))
        assert state.current == 0
        assert state.longest == 0

    def test_consecutive_weeks_build_a_streak(self):
        buckets = _weeks(0, 1, 2, 3)
        # today is inside week 3 (the last active week)
        state = streak_state(buckets, qualifies_active, date(2026, 1, 28))
        assert state.current == 4
        assert state.longest == 4
        assert state.in_progress is False

    def test_a_missed_week_ends_the_run(self):
        # weeks 0,1 then a gap at 2, then 3,4
        buckets = _weeks(0, 1, 3, 4)
        state = streak_state(buckets, qualifies_active, date(2026, 2, 2))
        assert state.current == 2
        assert state.longest == 2

    def test_current_week_not_yet_ridden_keeps_the_streak_alive(self):
        """Visiting on a Tuesday must not report a streak as broken."""
        buckets = _weeks(0, 1, 2)
        today = date(2026, 1, 5) + timedelta(weeks=3, days=1)  # Tuesday of week 3
        state = streak_state(buckets, qualifies_active, today)
        assert state.current == 3
        assert state.in_progress is True

    def test_riding_in_the_current_week_clears_in_progress(self):
        buckets = _weeks(0, 1, 2, 3)
        today = date(2026, 1, 5) + timedelta(weeks=3, days=1)
        state = streak_state(buckets, qualifies_active, today)
        assert state.current == 4
        assert state.in_progress is False

    def test_streak_is_broken_once_a_full_week_is_missed(self):
        """Grace ends with the week: two weeks off is a broken streak, not a pause."""
        buckets = _weeks(0, 1, 2)
        today = date(2026, 1, 5) + timedelta(weeks=4, days=1)
        state = streak_state(buckets, qualifies_active, today)
        assert state.current == 0
        assert state.longest == 3

    def test_longest_survives_a_later_break(self):
        buckets = _weeks(0, 1, 2, 3, 4, 6)
        state = streak_state(buckets, qualifies_active, date(2026, 2, 16))
        assert state.longest == 5
        assert state.current == 1

    def test_volume_predicate_needs_the_hours(self):
        under = [PeriodBucket(start=date(2026, 1, 5), count=1, seconds=4 * 3600)]
        over = [PeriodBucket(start=date(2026, 1, 5), count=1, seconds=6 * 3600)]
        assert streak_state(under, qualifies_volume, date(2026, 1, 8)).current == 0
        assert streak_state(over, qualifies_volume, date(2026, 1, 8)).current == 1

    def test_multisport_predicate_needs_two_sports(self):
        one = [PeriodBucket(start=date(2026, 1, 5), count=2, sports=frozenset({"cycling"}))]
        two = [
            PeriodBucket(
                start=date(2026, 1, 5), count=2, sports=frozenset({"cycling", "running"})
            )
        ]
        assert streak_state(one, qualifies_multisport, date(2026, 1, 8)).current == 0
        assert streak_state(two, qualifies_multisport, date(2026, 1, 8)).current == 1

    def test_monthly_streak_walks_months(self):
        buckets = [
            PeriodBucket(start=date(2026, 1, 1), count=1),
            PeriodBucket(start=date(2026, 2, 1), count=1),
            PeriodBucket(start=date(2026, 3, 1), count=1),
        ]
        state = streak_state(buckets, qualifies_active, date(2026, 3, 15), monthly=True)
        assert state.current == 3

    def test_monthly_streak_crosses_a_year_boundary(self):
        buckets = [
            PeriodBucket(start=date(2025, 11, 1), count=1),
            PeriodBucket(start=date(2025, 12, 1), count=1),
            PeriodBucket(start=date(2026, 1, 1), count=1),
        ]
        state = streak_state(buckets, qualifies_active, date(2026, 1, 20), monthly=True)
        assert state.current == 3


class TestStreakTierDates:
    def test_tier_earned_on_the_last_day_of_the_nth_week(self):
        buckets = _weeks(0, 1, 2, 3)
        today = date(2026, 2, 28)
        reached = streak_tier_dates(buckets, qualifies_active, [4], today)
        # week 3 runs Mon 2026-01-26 → Sun 2026-02-01
        assert reached[4] == date(2026, 2, 1)

    def test_unreached_tiers_are_absent(self):
        buckets = _weeks(0, 1)
        reached = streak_tier_dates(buckets, qualifies_active, [4, 8], date(2026, 1, 20))
        assert reached == {}

    def test_earned_date_never_lands_in_the_future(self):
        buckets = _weeks(0, 1, 2, 3)
        today = date(2026, 1, 28)  # mid-way through the 4th week
        reached = streak_tier_dates(buckets, qualifies_active, [4], today)
        assert reached[4] == today

    def test_earliest_qualifying_run_wins(self):
        """A later, longer streak must not re-date an already-earned tier."""
        buckets = _weeks(0, 1, 2, 3, 5, 6, 7, 8, 9)
        reached = streak_tier_dates(buckets, qualifies_active, [4], date(2026, 4, 1))
        assert reached[4] == date(2026, 2, 1)  # end of the first run's 4th week


# ── Comeback ─────────────────────────────────────────────────────────────────


class TestComeback:
    def test_no_gap_means_no_comeback(self):
        days = [date(2026, 1, 1), date(2026, 1, 5), date(2026, 1, 9)]
        assert comeback_date(days) is None

    def test_returns_the_day_training_resumed(self):
        days = [date(2026, 1, 1), date(2026, 3, 1)]
        assert comeback_date(days) == date(2026, 3, 1)

    def test_gap_must_reach_the_threshold(self):
        short = [date(2026, 1, 1), date(2026, 1, 1) + timedelta(days=COMEBACK_GAP_DAYS - 1)]
        exact = [date(2026, 1, 1), date(2026, 1, 1) + timedelta(days=COMEBACK_GAP_DAYS)]
        assert comeback_date(short) is None
        assert comeback_date(exact) is not None

    def test_earliest_comeback_is_kept(self):
        days = [date(2026, 1, 1), date(2026, 3, 1), date(2026, 8, 1)]
        assert comeback_date(days) == date(2026, 3, 1)

    def test_single_activity_has_no_comeback(self):
        assert comeback_date([date(2026, 1, 1)]) is None
        assert comeback_date([]) is None


@pytest.mark.parametrize("day", [date(2026, 1, 1), date(2026, 6, 15), date(2026, 12, 31)])
def test_week_start_is_always_a_monday(day):
    assert week_start(day).weekday() == 0
