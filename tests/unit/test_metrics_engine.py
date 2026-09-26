"""
Unit tests for backend/app/services/metrics_engine.py.

Uses the async session fixture from conftest.py to run against an in-memory DB.
"""
import math
import uuid
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import delete, select

from backend.app.models.user_orm import Activity, ActivitySource, Athlete, DailyMetric
from backend.app.services.metrics_engine import catch_up_metrics, recalculate_from

# EMA decay constants (same as production)
K42 = 1 - math.exp(-1 / 42)
K7 = 1 - math.exp(-1 / 7)

TODAY = date.today()


async def _make_athlete(session) -> Athlete:
    """Create a minimal Athlete in the team test DB and return it."""
    athlete = Athlete(
        id=str(uuid.uuid4()),
        global_user_id=str(uuid.uuid4()),
        ftp_tests=[],
    )
    session.add(athlete)
    await session.flush()
    return athlete


async def _make_activity(session, athlete_id: str, load: float, day: date) -> Activity:
    """Insert a processed Activity with the given Load on the given date."""
    activity = Activity(
        id=str(uuid.uuid4()),
        athlete_id=athlete_id,
        load=load,
        status="processed",
        start_time=datetime.combine(day, datetime.min.time()).replace(tzinfo=timezone.utc),
    )
    session.add(activity)
    await session.flush()
    session.add(ActivitySource(activity_id=activity.id, provider="manual"))
    await session.flush()
    return activity


async def _metrics_by_date(session, athlete_id: str) -> dict[date, DailyMetric]:
    result = await session.execute(
        select(DailyMetric).where(DailyMetric.athlete_id == athlete_id)
    )
    # Detached copies, so later writes to the same rows don't change them.
    return {
        m.date: DailyMetric(
            athlete_id=m.athlete_id, date=m.date, fitness=m.fitness,
            fatigue=m.fatigue, form=m.form, load_day=m.load_day,
        )
        for m in result.scalars()
    }


async def _delete_metrics_from(session, athlete_id: str, day: date) -> None:
    """Drop every DailyMetric row on or after ``day`` — the state of an athlete
    whose metrics were last written the day before."""
    await session.execute(
        delete(DailyMetric).where(
            DailyMetric.athlete_id == athlete_id,
            DailyMetric.date >= day,
        )
    )
    await session.flush()


class TestRecalculateFrom:
    async def test_single_activity_ctl_atl(self, session):
        athlete = await _make_athlete(session)
        await _make_activity(session, athlete.id, load=100.0, day=TODAY)

        await recalculate_from(athlete.id, TODAY, session)

        result = await session.execute(
            select(DailyMetric).where(
                DailyMetric.athlete_id == athlete.id,
                DailyMetric.date == TODAY,
            )
        )
        metric = result.scalar_one()

        assert metric.fitness == pytest.approx(100 * K42, rel=1e-6)
        assert metric.fatigue == pytest.approx(100 * K7, rel=1e-6)
        # Form is computed from yesterday's Fitness - Fatigue (both zero for first day)
        assert metric.form == pytest.approx(0.0, abs=1e-9)
        assert metric.load_day == pytest.approx(100.0, rel=1e-6)

    async def test_two_activities_same_day_tss_summed(self, session):
        athlete = await _make_athlete(session)
        await _make_activity(session, athlete.id, load=60.0, day=TODAY)
        await _make_activity(session, athlete.id, load=40.0, day=TODAY)

        await recalculate_from(athlete.id, TODAY, session)

        result = await session.execute(
            select(DailyMetric).where(
                DailyMetric.athlete_id == athlete.id,
                DailyMetric.date == TODAY,
            )
        )
        metric = result.scalar_one()

        # Both activities' Load should be summed before the EMA step
        assert metric.load_day == pytest.approx(100.0, rel=1e-6)
        assert metric.fitness == pytest.approx(100 * K42, rel=1e-6)

    async def test_second_day_inherits_previous_ctl_atl(self, session):
        athlete = await _make_athlete(session)
        yesterday = TODAY - timedelta(days=1)
        await _make_activity(session, athlete.id, load=100.0, day=yesterday)

        await recalculate_from(athlete.id, yesterday, session)

        # Day 1 (yesterday)
        r1 = await session.execute(
            select(DailyMetric).where(
                DailyMetric.athlete_id == athlete.id,
                DailyMetric.date == yesterday,
            )
        )
        m1 = r1.scalar_one()
        assert m1.fitness == pytest.approx(100 * K42, rel=1e-6)
        assert m1.fatigue == pytest.approx(100 * K7, rel=1e-6)

        # Day 2 (today) — no activity, so load_day=0
        r2 = await session.execute(
            select(DailyMetric).where(
                DailyMetric.athlete_id == athlete.id,
                DailyMetric.date == TODAY,
            )
        )
        m2 = r2.scalar_one()
        expected_ctl2 = m1.fitness + (0.0 - m1.fitness) * K42
        expected_atl2 = m1.fatigue + (0.0 - m1.fatigue) * K7
        assert m2.fitness == pytest.approx(expected_ctl2, rel=1e-6)
        assert m2.fatigue == pytest.approx(expected_atl2, rel=1e-6)
        # Form on day 2 = day 1's Fitness - day 1's Fatigue
        assert m2.form == pytest.approx(m1.fitness - m1.fatigue, rel=1e-6)

    async def test_missing_previous_day_seeds_from_last_stored_day(self, session):
        """Regression: rows were written through the day before yesterday, nothing
        wrote yesterday's, and a ride synced today. The recalculation used to find
        no row for yesterday and seed 0.0/0.0, resetting Fitness to ~1."""
        athlete = await _make_athlete(session)
        start = TODAY - timedelta(days=30)
        for i in range(29):
            await _make_activity(session, athlete.id, load=50.0, day=start + timedelta(days=i))
        await recalculate_from(athlete.id, start, session)
        expected = await _metrics_by_date(session, athlete.id)

        yesterday = TODAY - timedelta(days=1)
        await _delete_metrics_from(session, athlete.id, yesterday)
        await _make_activity(session, athlete.id, load=44.0, day=TODAY)

        await recalculate_from(athlete.id, TODAY, session)

        rows = await _metrics_by_date(session, athlete.id)
        # The hole is filled rather than left for the chart to jump over.
        assert yesterday in rows
        assert rows[yesterday].fitness == pytest.approx(expected[yesterday].fitness, rel=1e-9)
        assert rows[yesterday].fatigue == pytest.approx(expected[yesterday].fatigue, rel=1e-9)
        # Today continues from yesterday, not from zero.
        y = rows[yesterday]
        assert rows[TODAY].fitness == pytest.approx(y.fitness + (44.0 - y.fitness) * K42, rel=1e-9)
        assert rows[TODAY].fatigue == pytest.approx(y.fatigue + (44.0 - y.fatigue) * K7, rel=1e-9)
        assert rows[TODAY].form == pytest.approx(y.fitness - y.fatigue, rel=1e-9)

    async def test_empty_athlete_produces_no_metrics(self, session):
        athlete = await _make_athlete(session)
        # No activities — recalculate from today still runs (creates metrics with 0 load)
        await recalculate_from(athlete.id, TODAY, session)

        result = await session.execute(
            select(DailyMetric).where(DailyMetric.athlete_id == athlete.id)
        )
        metrics = result.scalars().all()
        # One row for today with zeroed-out values
        assert len(metrics) == 1
        assert metrics[0].load_day == pytest.approx(0.0)
        assert metrics[0].fitness == pytest.approx(0.0)


class TestCatchUpMetrics:
    async def test_no_update_when_metrics_match_activities(self, session):
        """No recalculation when stored load_day matches actual activity Load."""
        athlete = await _make_athlete(session)
        await _make_activity(session, athlete.id, load=80.0, day=TODAY)
        await recalculate_from(athlete.id, TODAY, session)

        updated = await catch_up_metrics(athlete.id, session)

        assert updated is False

    async def test_recalculates_when_activity_deleted(self, session):
        """Detects stale load_day after an activity is hard-deleted without going
        through the API endpoint (which would normally trigger _bg_recalculate)."""
        athlete = await _make_athlete(session)
        yesterday = TODAY - timedelta(days=1)

        act1 = await _make_activity(session, athlete.id, load=60.0, day=yesterday)
        act2 = await _make_activity(session, athlete.id, load=40.0, day=yesterday)
        await _make_activity(session, athlete.id, load=50.0, day=TODAY)
        await recalculate_from(athlete.id, yesterday, session)

        # Verify initial state: both days correct
        r = await session.execute(
            select(DailyMetric).where(
                DailyMetric.athlete_id == athlete.id,
                DailyMetric.date == yesterday,
            )
        )
        assert r.scalar_one().load_day == pytest.approx(100.0)

        # Simulate out-of-band hard delete (bypasses the API delete endpoint)
        await session.delete(act2)
        await session.flush()

        # catch_up_metrics should detect the mismatch and fix it
        updated = await catch_up_metrics(athlete.id, session)

        assert updated is True
        r2 = await session.execute(
            select(DailyMetric).where(
                DailyMetric.athlete_id == athlete.id,
                DailyMetric.date == yesterday,
            )
        )
        fixed = r2.scalar_one()
        # load_day should now reflect only act1's 60 Load
        assert fixed.load_day == pytest.approx(60.0)

    async def test_recalculates_cascade_from_earliest_stale_day(self, session):
        """When the stale day is before the forward-fill gap, recalculation
        starts from the stale day so subsequent days are also corrected."""
        athlete = await _make_athlete(session)
        two_days_ago = TODAY - timedelta(days=2)
        yesterday = TODAY - timedelta(days=1)

        act = await _make_activity(session, athlete.id, load=100.0, day=two_days_ago)
        await _make_activity(session, athlete.id, load=50.0, day=yesterday)
        # Don't create today's metric — forward fill gap exists
        await recalculate_from(athlete.id, two_days_ago, session)

        # Delete today's metric to simulate the forward-fill gap scenario
        r = await session.execute(
            select(DailyMetric).where(
                DailyMetric.athlete_id == athlete.id,
                DailyMetric.date == TODAY,
            )
        )
        today_metric = r.scalar_one()
        await session.delete(today_metric)
        await session.flush()

        # Also hard-delete the activity from two days ago (stale day is earlier)
        await session.delete(act)
        await session.flush()

        updated = await catch_up_metrics(athlete.id, session)

        assert updated is True
        r3 = await session.execute(
            select(DailyMetric).where(
                DailyMetric.athlete_id == athlete.id,
                DailyMetric.date == two_days_ago,
            )
        )
        fixed = r3.scalar_one()
        # Activity deleted → load_day should now be 0
        assert fixed.load_day == pytest.approx(0.0)

    async def test_heals_a_reset_left_across_a_missing_day(self, session):
        """An athlete already hit by the old seed-from-zero bug has a hole
        yesterday and a today row computed from 0.0/0.0. Today's row exists and
        its load_day is right, so neither the forward fill nor the load check
        saw anything; the hole itself must trigger the recalculation."""
        athlete = await _make_athlete(session)
        start = TODAY - timedelta(days=30)
        for i in range(29):
            await _make_activity(session, athlete.id, load=50.0, day=start + timedelta(days=i))
        await _make_activity(session, athlete.id, load=44.0, day=TODAY)
        await recalculate_from(athlete.id, start, session)
        expected = await _metrics_by_date(session, athlete.id)

        # Recreate the damage: yesterday missing, today seeded from zero.
        yesterday = TODAY - timedelta(days=1)
        await _delete_metrics_from(session, athlete.id, yesterday)
        session.add(DailyMetric(
            athlete_id=athlete.id, date=TODAY,
            fitness=44.0 * K42, fatigue=44.0 * K7, form=0.0, load_day=44.0,
        ))
        await session.flush()

        updated = await catch_up_metrics(athlete.id, session)

        assert updated is True
        rows = await _metrics_by_date(session, athlete.id)
        assert yesterday in rows
        for day in (yesterday, TODAY):
            assert rows[day].fitness == pytest.approx(expected[day].fitness, rel=1e-9)
            assert rows[day].fatigue == pytest.approx(expected[day].fatigue, rel=1e-9)
            assert rows[day].form == pytest.approx(expected[day].form, rel=1e-9)
        # Healed state is stable.
        assert await catch_up_metrics(athlete.id, session) is False

    async def test_back_fills_every_missing_day_across_the_whole_history(self, session):
        """Holes are back-filled however old they are — including ones outside
        the 90-day stale-Load window — and every day after the first hole is
        recomputed from real values rather than the zero seed it once got."""
        athlete = await _make_athlete(session)
        start = TODAY - timedelta(days=200)
        for i in range(0, 200, 2):
            await _make_activity(session, athlete.id, load=60.0, day=start + timedelta(days=i))
        await recalculate_from(athlete.id, start, session)
        expected = await _metrics_by_date(session, athlete.id)

        # Two holes: one well outside the stale window, one yesterday. Every
        # row after the old hole carries the zero-seeded values the old code
        # wrote; their load_day is still right, so only the hole gives it away.
        old_hole = TODAY - timedelta(days=150)
        recent_hole = TODAY - timedelta(days=1)
        for hole in (old_hole, recent_hole):
            await session.execute(
                delete(DailyMetric).where(
                    DailyMetric.athlete_id == athlete.id, DailyMetric.date == hole,
                )
            )
        rows = await session.execute(
            select(DailyMetric).where(
                DailyMetric.athlete_id == athlete.id, DailyMetric.date > old_hole,
            )
        )
        for m in rows.scalars():
            m.fitness = m.fatigue = m.form = 0.0
        await session.flush()

        assert await catch_up_metrics(athlete.id, session) is True

        healed = await _metrics_by_date(session, athlete.id)
        assert healed.keys() == expected.keys()
        for day, m in expected.items():
            assert healed[day].fitness == pytest.approx(m.fitness, rel=1e-9), day
            assert healed[day].fatigue == pytest.approx(m.fatigue, rel=1e-9), day
            assert healed[day].form == pytest.approx(m.form, rel=1e-9), day
        assert await catch_up_metrics(athlete.id, session) is False
