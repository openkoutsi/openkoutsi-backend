"""Re-fitting CP/W' snapshots after a provider backlog import (issue #77).

A provider sync walks pages newest-first, while the CP fit only looks *backwards*
in time from each ride's own date. During the walk, every ride already stored is
newer than the one being processed and so excluded from its fit — meaning every
ride in a full-history import is fit against a single ride's efforts, and freezes
that way permanently.

These tests drive that ordering directly rather than mocking a provider, so the
defect and the repair are both visible in the assertions.
"""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from backend.app.models.user_orm import (
    Activity,
    ActivityPowerBest,
    ActivityStream,
    Athlete,
)
from backend.app.services.aerobic_metrics import refit_cp_snapshots
from backend.app.services.power_profile import cp_wprime_as_of

_BASE = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)

# Bests that fit exactly to CP = 195 W, W' = 15 kJ: W(t) = 195·t + 15000.
_GOOD_BESTS = {120: 320.0, 180: 278.3333, 300: 245.0, 480: 226.25, 900: 211.6667, 1200: 207.5}

# A weaker season: CP = 170 W, W' = 12 kJ.
_OLD_BESTS = {120: 270.0, 180: 236.6667, 300: 210.0, 480: 195.0, 900: 183.3333, 1200: 180.0}


async def _make_athlete(session) -> Athlete:
    athlete = Athlete(global_user_id="user-refit", ftp=250, ftp_tests=[])
    session.add(athlete)
    await session.commit()
    await session.refresh(athlete)
    return athlete


async def _add_ride(
    session, athlete: Athlete, *, days_ago: int, bests: dict[int, float],
    with_power_stream: bool = True,
) -> Activity:
    """One ride with its power bests, and optionally a power stream."""
    start = _BASE - timedelta(days=days_ago)
    activity = Activity(
        athlete_id=athlete.id,
        name=f"Ride -{days_ago}d",
        sport_type="Ride",
        start_time=start,
        duration_s=4000,
        avg_power=200.0,
        weighted_power=206.0,
        avg_hr=140.0,
        status="processed",
    )
    session.add(activity)
    await session.flush()

    for duration_s, power_w in bests.items():
        session.add(ActivityPowerBest(
            activity_id=activity.id,
            athlete_id=athlete.id,
            duration_s=duration_s,
            power_w=power_w,
            activity_start_time=start,
        ))
    if with_power_stream:
        session.add(ActivityStream(
            activity_id=activity.id,
            stream_type="power",
            data=[300.0] * 120 + [180.0] * 3880,
        ))
    await session.commit()
    return activity


async def _simulate_backlog_import(
    session, athlete: Athlete, ride_specs: list[tuple[int, dict[int, float]]],
    *, with_power_stream: bool = True,
) -> list[Activity]:
    """Import rides newest-first, fitting each one *as it lands*.

    The interleaving is the whole point. A provider walk creates a ride, writes
    its power bests, and fits it before moving to the next — so each fit sees
    only the rides already imported, all of which are newer and therefore
    excluded by the as-of restriction. Populating every ride's bests up front and
    fitting afterwards would quietly test the repaired world instead of the
    broken one.

    ``ride_specs`` is ``[(days_ago, bests), …]``; they are imported in
    newest-first order regardless of how they are listed.
    """
    created: list[Activity] = []
    for days_ago, bests in sorted(ride_specs, key=lambda spec: spec[0]):
        activity = await _add_ride(
            session, athlete, days_ago=days_ago, bests=bests,
            with_power_stream=with_power_stream,
        )
        cp, w_prime, points = await cp_wprime_as_of(
            athlete.id, session, activity.start_time
        )
        activity.cp_w = round(cp, 1) if cp is not None else None
        activity.w_prime_j = round(w_prime) if w_prime is not None else None
        activity.cp_fit_points = points
        created.append(activity)
    await session.commit()
    # Oldest first, which is how the callers below want to refer to them.
    return sorted(created, key=lambda a: a.start_time)


class TestRefitCpSnapshots:
    async def test_backlog_import_leaves_thin_fits_that_the_refit_repairs(
        self, session
    ):
        athlete = await _make_athlete(session)
        # A strong early season, then two weaker later rides. Every ride is fit
        # against its own efforts alone during the newest-first walk.
        oldest, middle, newest = await _simulate_backlog_import(session, athlete, [
            (360, _GOOD_BESTS),
            (180, _OLD_BESTS),
            (0, _OLD_BESTS),
        ])

        # The defect: the two later rides never saw the strong early season, so
        # both were fit from their own (weaker) efforts.
        assert newest.cp_w == pytest.approx(170.0, abs=1.0)
        assert middle.cp_w == pytest.approx(170.0, abs=1.0)
        assert oldest.cp_w == pytest.approx(195.0, abs=1.0)

        changed = await refit_cp_snapshots(
            athlete.id, session, since=oldest.start_time
        )
        await session.commit()
        assert changed == 2, "both later rides should be corrected"

        # Both now see the earlier, stronger efforts they should always have
        # been judged against.
        await session.refresh(newest)
        await session.refresh(middle)
        assert newest.cp_w == pytest.approx(195.0, abs=1.0)
        assert middle.cp_w == pytest.approx(195.0, abs=1.0)

        # The oldest ride is unchanged: nothing precedes it.
        await session.refresh(oldest)
        assert oldest.cp_w == pytest.approx(195.0, abs=1.0)

    async def test_a_later_ride_fit_before_older_history_arrived_is_corrected(
        self, session
    ):
        """The case that actually changes a number, not just the point count."""
        athlete = await _make_athlete(session)
        # A strong older ride, and a weaker recent one imported first.
        older, recent = await _simulate_backlog_import(session, athlete, [
            (200, _GOOD_BESTS),
            (0, _OLD_BESTS),
        ])

        # During the walk, `recent` was processed first and saw only its own
        # (weaker) efforts — so it got the weaker CP.
        fit_during_import = recent.cp_w
        assert fit_during_import == pytest.approx(170.0, abs=1.0)

        await refit_cp_snapshots(athlete.id, session, since=older.start_time)
        await session.commit()

        # With the older, stronger ride now visible, the recent ride's as-of fit
        # rises to reflect the profile the athlete actually had.
        await session.refresh(recent)
        assert recent.cp_w == pytest.approx(195.0, abs=1.0)
        assert recent.cp_w > fit_during_import

    async def test_older_rides_are_not_given_anachronistic_fits(self, session):
        """The re-fit must not undo the 'as of' restriction it exists to serve."""
        athlete = await _make_athlete(session)
        old = await _add_ride(session, athlete, days_ago=300, bests=_OLD_BESTS)
        await _add_ride(session, athlete, days_ago=0, bests=_GOOD_BESTS)

        await refit_cp_snapshots(athlete.id, session, since=old.start_time)
        await session.commit()

        # The old ride keeps the weaker CP: the newer, stronger efforts postdate
        # it and must stay excluded.
        await session.refresh(old)
        assert old.cp_w == pytest.approx(170.0, abs=1.0)

    async def test_scoped_to_since_leaves_earlier_activities_untouched(self, session):
        athlete = await _make_athlete(session)
        untouched = await _add_ride(session, athlete, days_ago=400, bests=_OLD_BESTS)
        untouched.cp_w = 999.0
        untouched.w_prime_j = 999
        untouched.cp_fit_points = 99
        recent = await _add_ride(session, athlete, days_ago=10, bests=_GOOD_BESTS)
        await session.commit()

        await refit_cp_snapshots(athlete.id, session, since=recent.start_time)
        await session.commit()

        # An incremental sync must not re-fit the whole history.
        await session.refresh(untouched)
        assert untouched.cp_w == 999.0
        assert untouched.cp_fit_points == 99

    async def test_w_bal_stream_is_rebuilt_when_the_fit_changes(self, session):
        athlete = await _make_athlete(session)
        older, recent = await _simulate_backlog_import(session, athlete, [
            (200, _GOOD_BESTS),
            (0, _OLD_BESTS),
        ])

        await refit_cp_snapshots(athlete.id, session, since=older.start_time)
        await session.commit()

        streams = (await session.execute(
            select(ActivityStream).where(
                ActivityStream.activity_id == recent.id,
                ActivityStream.stream_type == "w_bal",
            )
        )).scalars().all()
        assert len(streams) == 1, "exactly one w_bal row, not a duplicate per re-fit"

        await session.refresh(recent)
        assert len(streams[0].data) == 4000
        assert max(streams[0].data) <= recent.w_prime_j

    async def test_repeated_refits_are_idempotent(self, session):
        athlete = await _make_athlete(session)
        older, recent = await _simulate_backlog_import(session, athlete, [
            (200, _GOOD_BESTS),
            (0, _OLD_BESTS),
        ])

        first = await refit_cp_snapshots(athlete.id, session, since=older.start_time)
        await session.commit()
        second = await refit_cp_snapshots(athlete.id, session, since=older.start_time)
        await session.commit()

        assert first > 0
        assert second == 0, "a converged re-fit must report nothing changed"

        streams = (await session.execute(
            select(ActivityStream).where(
                ActivityStream.activity_id == recent.id,
                ActivityStream.stream_type == "w_bal",
            )
        )).scalars().all()
        assert len(streams) == 1

    async def test_activity_without_a_power_stream_gets_no_w_bal(self, session):
        athlete = await _make_athlete(session)
        older, recent = await _simulate_backlog_import(
            session, athlete,
            [(200, _GOOD_BESTS), (0, _OLD_BESTS)],
            with_power_stream=False,
        )

        await refit_cp_snapshots(athlete.id, session, since=older.start_time)
        await session.commit()

        streams = (await session.execute(
            select(ActivityStream).where(
                ActivityStream.activity_id == recent.id,
                ActivityStream.stream_type == "w_bal",
            )
        )).scalars().all()
        assert streams == []
        # The snapshot columns are still corrected — only the stream needs power.
        await session.refresh(recent)
        assert recent.cp_w is not None

    async def test_no_activities_in_range_is_a_no_op(self, session):
        athlete = await _make_athlete(session)
        await _add_ride(session, athlete, days_ago=300, bests=_GOOD_BESTS)
        changed = await refit_cp_snapshots(
            athlete.id, session, since=_BASE + timedelta(days=1)
        )
        assert changed == 0

    async def test_matches_the_per_activity_query_path(self, session):
        """The incremental walk must agree with `cp_wprime_as_of` exactly.

        The re-fit maintains a running rank-1 maximum instead of querying per
        activity, so the two implementations have to be checked against each
        other or they will drift.
        """
        athlete = await _make_athlete(session)
        rides = [
            await _add_ride(session, athlete, days_ago=d, bests=b)
            for d, b in ((360, _OLD_BESTS), (200, _GOOD_BESTS), (30, _OLD_BESTS))
        ]

        await refit_cp_snapshots(athlete.id, session, since=rides[0].start_time)
        await session.commit()

        for activity in rides:
            cp, w_prime, points = await cp_wprime_as_of(
                athlete.id, session, activity.start_time
            )
            await session.refresh(activity)
            assert activity.cp_w == (round(cp, 1) if cp is not None else None)
            assert activity.w_prime_j == (round(w_prime) if w_prime is not None else None)
            assert activity.cp_fit_points == points
