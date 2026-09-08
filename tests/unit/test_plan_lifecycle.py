"""Tests for the plan auto-close pass (``services.plan_lifecycle``).

A plan closes itself once its last scheduled day has passed. What these cover is
the boundary (the end date itself is still part of the plan), whose day decides
it (the athlete's, not the server's), and the two ways the pass must not fight
the athlete: it never reopens what it closed, and it leaves a reopened plan open.
"""
from datetime import date, timedelta

import pytest

from backend.app.models.user_orm import TrainingPlan
from backend.app.services.plan_lifecycle import close_finished_plans

_START = date(2025, 6, 2)  # A Monday
_END = _START + timedelta(weeks=4) - timedelta(days=1)  # inclusive last day


async def _plan(session, athlete_id, *, status="active", start=_START, end=_END,
                completed_at=None, name="P"):
    plan = TrainingPlan(
        athlete_id=athlete_id, name=name, start_date=start, end_date=end,
        status=status, completed_at=completed_at,
    )
    session.add(plan)
    await session.commit()
    return plan


class TestCloseFinishedPlans:
    async def test_closes_a_plan_whose_last_day_has_passed(self, session, seeded_athlete):
        plan = await _plan(session, seeded_athlete.id)

        closed = await close_finished_plans(
            seeded_athlete.id, session, today=_END + timedelta(days=1)
        )

        assert [p.id for p in closed] == [plan.id]
        assert plan.status == "completed"
        assert plan.completed_at is not None

    async def test_end_date_itself_is_still_part_of_the_plan(self, session, seeded_athlete):
        plan = await _plan(session, seeded_athlete.id)

        assert await close_finished_plans(seeded_athlete.id, session, today=_END) == []
        assert plan.status == "active"
        assert plan.completed_at is None

    async def test_running_plan_is_left_alone(self, session, seeded_athlete):
        plan = await _plan(session, seeded_athlete.id)

        assert await close_finished_plans(
            seeded_athlete.id, session, today=_START + timedelta(days=3)
        ) == []
        assert plan.status == "active"

    async def test_open_ended_plan_never_closes(self, session, seeded_athlete):
        # No end date: there is no last day to be past, and inferring one from
        # `weeks` would close plans whose length was never recorded.
        plan = await _plan(session, seeded_athlete.id, end=None)

        assert await close_finished_plans(
            seeded_athlete.id, session, today=_END + timedelta(days=365)
        ) == []
        assert plan.status == "active"

    async def test_archived_plan_is_left_alone(self, session, seeded_athlete):
        plan = await _plan(session, seeded_athlete.id, status="archived")

        assert await close_finished_plans(
            seeded_athlete.id, session, today=_END + timedelta(days=1)
        ) == []
        assert plan.status == "archived"
        assert plan.completed_at is None

    async def test_is_idempotent(self, session, seeded_athlete):
        plan = await _plan(session, seeded_athlete.id)
        today = _END + timedelta(days=1)

        await close_finished_plans(seeded_athlete.id, session, today=today)
        stamp = plan.completed_at

        assert await close_finished_plans(seeded_athlete.id, session, today=today) == []
        assert plan.completed_at == stamp

    async def test_reopened_plan_is_not_closed_again(self, session, seeded_athlete):
        """The athlete's reopen wins: `completed_at` is what says "already had
        its turn", so putting the plan back to active does not invite the next
        read to shut it again."""
        plan = await _plan(session, seeded_athlete.id)
        today = _END + timedelta(days=1)
        await close_finished_plans(seeded_athlete.id, session, today=today)

        plan.status = "active"  # what POST /plans/{id}/unarchive does
        await session.commit()

        assert await close_finished_plans(seeded_athlete.id, session, today=today) == []
        assert plan.status == "active"

    async def test_closes_only_this_athletes_plans(self, session, seeded_athlete):
        from backend.app.models.user_orm import Athlete

        other = Athlete(id="other-athlete", global_user_id="other-user", ftp_tests=[])
        session.add(other)
        await session.commit()
        mine = await _plan(session, seeded_athlete.id, name="Mine")
        theirs = await _plan(session, other.id, name="Theirs")

        closed = await close_finished_plans(
            seeded_athlete.id, session, today=_END + timedelta(days=1)
        )

        assert [p.id for p in closed] == [mine.id]
        assert theirs.status == "active"

    @pytest.mark.parametrize(
        "timezone_name,expected_status",
        [
            # 13:00 UTC on the plan's last day. In Auckland (UTC+13) it is
            # already the next day, so the plan is over; in Los Angeles
            # (UTC-7) it is still the last day, and still the plan's.
            ("Pacific/Auckland", "completed"),
            ("America/Los_Angeles", "active"),
        ],
    )
    async def test_boundary_follows_the_athletes_local_day(
        self, session, seeded_athlete, monkeypatch, timezone_name, expected_status
    ):
        from datetime import datetime, timezone as dt_timezone

        import backend.app.services.plan_lifecycle as lifecycle

        seeded_athlete.app_settings = {"timezone": timezone_name}
        plan = await _plan(session, seeded_athlete.id)

        instant = datetime(_END.year, _END.month, _END.day, 13, 0, tzinfo=dt_timezone.utc)

        class _FixedDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return instant.astimezone(tz) if tz else instant

        monkeypatch.setattr(lifecycle, "datetime", _FixedDatetime)

        # No explicit `today`: the athlete's own zone has to decide it.
        await close_finished_plans(seeded_athlete.id, session, athlete=seeded_athlete)

        assert plan.status == expected_status

    async def test_athlete_is_loaded_when_not_supplied(self, session, seeded_athlete):
        """The ingest paths pass only an id, so the pass has to find the athlete
        itself to know whose day it is."""
        plan = await _plan(session, seeded_athlete.id)

        closed = await close_finished_plans(
            seeded_athlete.id, session, today=_END + timedelta(days=1)
        )

        assert [p.id for p in closed] == [plan.id]
