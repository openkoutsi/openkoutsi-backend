"""Unit tests for the daily achievement sweep (issue #69).

The write paths only mark now, so something has to settle. ``GET /achievements``
does for anyone who opens the app; this sweep is what stops an athlete who
uploads from a head unit and never looks from having their inbox message wait
indefinitely — and arrive dated whenever they happened to look rather than near
when the badge was earned.

The properties that matter: it settles what is marked, it leaves alone what is
not, and one broken user database does not cost every other athlete their sweep.
"""
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from backend.app.models.registry_orm import User
from backend.app.models.user_orm import AchievementUnlock, Activity
from backend.app.services import achievements_sweep

_TEST_USER_ID = "test-user-00000000"
_TODAY = date(2026, 7, 22)


@pytest.fixture
async def sweep_session(registry_engine, registry_session):
    """A second session on the same in-memory registry as ``registry_session``."""
    factory = async_sessionmaker(registry_engine, expire_on_commit=False)
    async with factory() as s:
        yield s


@pytest.fixture
def user_factory(user_engine):
    """Point the sweep's per-user lookup at this test's in-memory user DB."""
    factory = async_sessionmaker(user_engine, expire_on_commit=False)
    with patch(
        "backend.app.db.user_session.get_user_session_factory",
        return_value=factory,
    ) as mock:
        yield mock


@pytest.fixture(autouse=True)
def _no_inbox():
    """The inbox write opens its own session against a real file; assert on it here."""
    with patch(
        "backend.app.services.achievements.notify_user", AsyncMock()
    ) as mock:
        yield mock


async def _ride(session, athlete, *, day=_TODAY):
    session.add(
        Activity(
            athlete_id=athlete.id,
            sport_type="Ride",
            start_time=datetime(day.year, day.month, day.day, 10, tzinfo=timezone.utc),
            status="processed",
            duration_s=3600,
        )
    )
    await session.commit()


async def _unlock_count(session, athlete) -> int:
    rows = (
        await session.execute(
            select(AchievementUnlock).where(
                AchievementUnlock.athlete_id == athlete.id
            )
        )
    ).scalars().all()
    return len(rows)


class TestSweep:
    async def test_a_marked_athlete_is_settled(
        self, sweep_session, session, seeded_athlete, user_factory, _no_inbox
    ):
        await _ride(session, seeded_athlete)
        seeded_athlete.achievements_dirty_at = datetime.now(timezone.utc)
        await session.commit()

        settled = await achievements_sweep.run_achievements_sweep(sweep_session)

        assert settled == 1
        await session.refresh(seeded_athlete)
        assert seeded_athlete.achievements_dirty_at is None
        assert await _unlock_count(session, seeded_athlete) > 0
        # The whole reason the sweep exists: the athlete never opened the app.
        _no_inbox.assert_awaited_once()

    async def test_a_clean_athlete_is_left_alone(
        self, sweep_session, session, seeded_athlete, user_factory, _no_inbox
    ):
        """No mark means no debt — and no full-history scan to pay for it."""
        await _ride(session, seeded_athlete)
        assert seeded_athlete.achievements_dirty_at is None

        settled = await achievements_sweep.run_achievements_sweep(sweep_session)

        assert settled == 0
        assert await _unlock_count(session, seeded_athlete) == 0
        _no_inbox.assert_not_awaited()

    async def test_the_sweep_is_idempotent(
        self, sweep_session, session, seeded_athlete, user_factory, _no_inbox
    ):
        """Clearing the mark is what stops a daily re-announcement."""
        await _ride(session, seeded_athlete)
        seeded_athlete.achievements_dirty_at = datetime.now(timezone.utc)
        await session.commit()

        assert await achievements_sweep.run_achievements_sweep(sweep_session) == 1
        assert await achievements_sweep.run_achievements_sweep(sweep_session) == 0
        _no_inbox.assert_awaited_once()

    async def test_one_broken_user_db_does_not_stop_the_rest(
        self, sweep_session, session, seeded_athlete, user_factory, _no_inbox
    ):
        """A user with no database at all looks exactly like this.

        ``_get_user_engine`` creates no file, so opening a session for someone
        who has none raises rather than bringing one into existence (issue #102).
        """
        await _ride(session, seeded_athlete)
        seeded_athlete.achievements_dirty_at = datetime.now(timezone.utc)
        await session.commit()

        sweep_session.add(
            User(
                id="user-with-no-db",
                username="ghost",
                password_hash="x",
                created_at=datetime.now(timezone.utc),
            )
        )
        await sweep_session.commit()

        real = achievements_sweep._settle_user

        async def flaky(user_id: str) -> int:
            if user_id != _TEST_USER_ID:
                raise RuntimeError("no such database")
            return await real(user_id)

        with patch.object(achievements_sweep, "_settle_user", flaky):
            settled = await achievements_sweep.run_achievements_sweep(sweep_session)

        # The good user was still swept.
        assert settled == 1
        await session.refresh(seeded_athlete)
        assert seeded_athlete.achievements_dirty_at is None

    async def test_a_deleted_user_is_skipped(
        self, sweep_session, session, seeded_athlete, user_factory, _no_inbox
    ):
        await _ride(session, seeded_athlete)
        seeded_athlete.achievements_dirty_at = datetime.now(timezone.utc)
        await session.commit()

        user = (
            await sweep_session.execute(select(User).where(User.id == _TEST_USER_ID))
        ).scalar_one()
        user.deleted_at = datetime.now(timezone.utc) - timedelta(days=1)
        await sweep_session.commit()

        assert await achievements_sweep.run_achievements_sweep(sweep_session) == 0
        _no_inbox.assert_not_awaited()
