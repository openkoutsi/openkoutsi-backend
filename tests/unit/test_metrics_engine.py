"""
Unit tests for backend/app/services/metrics_engine.py.

Uses the async session fixture from conftest.py to run against an in-memory DB.
"""
import math
import uuid
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from backend.app.models.team_orm import Activity, ActivitySource, Athlete, DailyMetric
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


async def _make_activity(session, athlete_id: str, tss: float, day: date) -> Activity:
    """Insert a processed Activity with the given TSS on the given date."""
    activity = Activity(
        id=str(uuid.uuid4()),
        athlete_id=athlete_id,
        tss=tss,
        status="processed",
        start_time=datetime.combine(day, datetime.min.time()).replace(tzinfo=timezone.utc),
    )
    session.add(activity)
    await session.flush()
    session.add(ActivitySource(activity_id=activity.id, provider="manual"))
    await session.flush()
    return activity


class TestRecalculateFrom:
    async def test_single_activity_ctl_atl(self, session):
        athlete = await _make_athlete(session)
        await _make_activity(session, athlete.id, tss=100.0, day=TODAY)

        await recalculate_from(athlete.id, TODAY, session)

        result = await session.execute(
            select(DailyMetric).where(
                DailyMetric.athlete_id == athlete.id,
                DailyMetric.date == TODAY,
            )
        )
        metric = result.scalar_one()

        assert metric.ctl == pytest.approx(100 * K42, rel=1e-6)
        assert metric.atl == pytest.approx(100 * K7, rel=1e-6)
        # TSB is computed from yesterday's CTL - ATL (both zero for first day)
        assert metric.tsb == pytest.approx(0.0, abs=1e-9)
        assert metric.tss_day == pytest.approx(100.0, rel=1e-6)

    async def test_two_activities_same_day_tss_summed(self, session):
        athlete = await _make_athlete(session)
        await _make_activity(session, athlete.id, tss=60.0, day=TODAY)
        await _make_activity(session, athlete.id, tss=40.0, day=TODAY)

        await recalculate_from(athlete.id, TODAY, session)

        result = await session.execute(
            select(DailyMetric).where(
                DailyMetric.athlete_id == athlete.id,
                DailyMetric.date == TODAY,
            )
        )
        metric = result.scalar_one()

        # Both activities' TSS should be summed before the EMA step
        assert metric.tss_day == pytest.approx(100.0, rel=1e-6)
        assert metric.ctl == pytest.approx(100 * K42, rel=1e-6)

    async def test_second_day_inherits_previous_ctl_atl(self, session):
        athlete = await _make_athlete(session)
        yesterday = TODAY - timedelta(days=1)
        await _make_activity(session, athlete.id, tss=100.0, day=yesterday)

        await recalculate_from(athlete.id, yesterday, session)

        # Day 1 (yesterday)
        r1 = await session.execute(
            select(DailyMetric).where(
                DailyMetric.athlete_id == athlete.id,
                DailyMetric.date == yesterday,
            )
        )
        m1 = r1.scalar_one()
        assert m1.ctl == pytest.approx(100 * K42, rel=1e-6)
        assert m1.atl == pytest.approx(100 * K7, rel=1e-6)

        # Day 2 (today) — no activity, so tss_day=0
        r2 = await session.execute(
            select(DailyMetric).where(
                DailyMetric.athlete_id == athlete.id,
                DailyMetric.date == TODAY,
            )
        )
        m2 = r2.scalar_one()
        expected_ctl2 = m1.ctl + (0.0 - m1.ctl) * K42
        expected_atl2 = m1.atl + (0.0 - m1.atl) * K7
        assert m2.ctl == pytest.approx(expected_ctl2, rel=1e-6)
        assert m2.atl == pytest.approx(expected_atl2, rel=1e-6)
        # TSB on day 2 = day 1's CTL - day 1's ATL
        assert m2.tsb == pytest.approx(m1.ctl - m1.atl, rel=1e-6)

    async def test_empty_athlete_produces_no_metrics(self, session):
        athlete = await _make_athlete(session)
        # No activities — recalculate from today still runs (creates metrics with 0 tss)
        await recalculate_from(athlete.id, TODAY, session)

        result = await session.execute(
            select(DailyMetric).where(DailyMetric.athlete_id == athlete.id)
        )
        metrics = result.scalars().all()
        # One row for today with zeroed-out values
        assert len(metrics) == 1
        assert metrics[0].tss_day == pytest.approx(0.0)
        assert metrics[0].ctl == pytest.approx(0.0)


class TestCatchUpMetrics:
    async def test_no_update_when_metrics_match_activities(self, session):
        """No recalculation when stored tss_day matches actual activity TSS."""
        athlete = await _make_athlete(session)
        await _make_activity(session, athlete.id, tss=80.0, day=TODAY)
        await recalculate_from(athlete.id, TODAY, session)

        updated = await catch_up_metrics(athlete.id, session)

        assert updated is False

    async def test_recalculates_when_activity_deleted(self, session):
        """Detects stale tss_day after an activity is hard-deleted without going
        through the API endpoint (which would normally trigger _bg_recalculate)."""
        athlete = await _make_athlete(session)
        yesterday = TODAY - timedelta(days=1)

        act1 = await _make_activity(session, athlete.id, tss=60.0, day=yesterday)
        act2 = await _make_activity(session, athlete.id, tss=40.0, day=yesterday)
        await _make_activity(session, athlete.id, tss=50.0, day=TODAY)
        await recalculate_from(athlete.id, yesterday, session)

        # Verify initial state: both days correct
        r = await session.execute(
            select(DailyMetric).where(
                DailyMetric.athlete_id == athlete.id,
                DailyMetric.date == yesterday,
            )
        )
        assert r.scalar_one().tss_day == pytest.approx(100.0)

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
        # tss_day should now reflect only act1's 60 TSS
        assert fixed.tss_day == pytest.approx(60.0)

    async def test_recalculates_cascade_from_earliest_stale_day(self, session):
        """When the stale day is before the forward-fill gap, recalculation
        starts from the stale day so subsequent days are also corrected."""
        athlete = await _make_athlete(session)
        two_days_ago = TODAY - timedelta(days=2)
        yesterday = TODAY - timedelta(days=1)

        act = await _make_activity(session, athlete.id, tss=100.0, day=two_days_ago)
        await _make_activity(session, athlete.id, tss=50.0, day=yesterday)
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
        # Activity deleted → tss_day should now be 0
        assert fixed.tss_day == pytest.approx(0.0)
