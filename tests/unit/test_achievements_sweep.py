"""Unit tests for the daily achievement sweep (issue #69).

The write paths only mark now, so something has to settle. ``GET /achievements``
does for anyone who opens the app; this sweep is what stops an athlete who
uploads from a head unit and never looks from having their inbox message wait
indefinitely — and arrive dated whenever they happened to look rather than near
when the badge was earned.

The properties that matter: it settles what is marked, it leaves alone what is
not, and one broken user database does not cost every other athlete their sweep.
"""
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from backend.app.core.config import settings
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
    """Point the sweep's per-user lookup at this test's in-memory user DB.

    The file is created as well as the patch installed: the sweep decides who to
    visit by whether a user's database exists on disk, and this user is standing
    in for one who has been through activation. The session still goes to the
    in-memory engine; the file only has to be there.
    """
    Path(settings.user_db_path(_TEST_USER_ID)).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.user_db_path(_TEST_USER_ID)).touch()
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


def _give_database(user_id: str) -> None:
    """Put a file where this user's database goes, so the sweep visits them."""
    path = Path(settings.user_db_path(user_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _add_user(sweep_session, user_id: str, username: str | None, **kw) -> None:
    sweep_session.add(
        User(
            id=user_id,
            username=username,
            password_hash="x",
            created_at=datetime.now(timezone.utc),
            **kw,
        )
    )


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
        """A database that exists but will not open costs only its own user."""
        await _ride(session, seeded_athlete)
        seeded_athlete.achievements_dirty_at = datetime.now(timezone.utc)
        await session.commit()

        _add_user(sweep_session, "user-with-broken-db", "ghost")
        await sweep_session.commit()
        # Has a file, so the sweep visits it; opening it is what fails.
        _give_database("user-with-broken-db")

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

    async def test_a_user_with_no_database_is_skipped_quietly(
        self, sweep_session, session, seeded_athlete, user_factory, _no_inbox, caplog
    ):
        """A pending signup is a live user with no database, not a failure.

        Signup writes the registry row; activation writes the database. Visiting
        one of these raised ``unable to open database file`` and logged a full
        traceback for it — every day, for every signup that was never confirmed.
        """
        await _ride(session, seeded_athlete)
        seeded_athlete.achievements_dirty_at = datetime.now(timezone.utc)
        await session.commit()

        # No `_give_database`: exactly what `POST /auth/signup` leaves behind.
        _add_user(sweep_session, "pending-signup", None, email="nobody@example.com")
        await sweep_session.commit()

        with caplog.at_level(logging.DEBUG, logger=achievements_sweep.log.name):
            settled = await achievements_sweep.run_achievements_sweep(sweep_session)

        # The real user was still swept...
        assert settled == 1
        await session.refresh(seeded_athlete)
        assert seeded_athlete.achievements_dirty_at is None

        # ...and the pending signup was passed over without an error.
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            r.levelno == logging.DEBUG and "pending-signup" in r.getMessage()
            for r in caplog.records
        )

    async def test_a_user_with_no_database_is_never_opened(
        self, sweep_session, session, seeded_athlete, user_factory, _no_inbox
    ):
        """The skip is what keeps an engine off the 256-slot cache, so it has to
        happen before ``_settle_user``, not inside its exception handler."""
        _add_user(sweep_session, "pending-signup", None, email="nobody@example.com")
        await sweep_session.commit()

        visited: list[str] = []

        async def record(user_id: str) -> int:
            visited.append(user_id)
            return 0

        with patch.object(achievements_sweep, "_settle_user", record):
            await achievements_sweep.run_achievements_sweep(sweep_session)

        assert visited == [_TEST_USER_ID]

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
