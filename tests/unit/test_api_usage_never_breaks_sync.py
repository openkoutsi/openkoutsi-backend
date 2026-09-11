"""Accounting must never fail a sync (issue #66).

The single most important property carried over from ``record_llm_usage``, and
the one worth its own file: a locked database or a full disk on the *usage* file
must cost an observability row, never an athlete's import.

This exercises it through the real sync pipeline rather than at the transport
seam, because that is where the cost of getting it wrong would actually land —
the pipeline is the thing whose failure an athlete would notice.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import backend.app.models.user_orm  # noqa: F401 — populate UserBase.metadata
from backend.app.db.base import UserBase
from backend.app.models.registry_orm import ProviderConnection
from backend.app.models.user_orm import Activity, Athlete
from backend.app.services import api_usage as api_usage_service
from backend.app.services.provider_sync import sync_provider_activities
from backend.app.services.providers.base import NormalizedActivity

_USER_ID = "user-66"


def _norm() -> NormalizedActivity:
    return NormalizedActivity(
        external_id="ext-1",
        source="strava",
        name="Morning Ride",
        sport_type="Ride",
        start_time=datetime.now(timezone.utc) - timedelta(days=1),
        duration_s=3600,
        distance_m=50_000.0,
        elevation_m=500.0,
        avg_power=None,
        avg_hr=None,
        max_hr=None,
        avg_speed_ms=14.0,
        avg_cadence=None,
    )


def _provider() -> MagicMock:
    client = MagicMock()
    client.list_activities = AsyncMock(side_effect=[[_norm()], []])
    client.download_fit_file = AsyncMock(side_effect=Exception("no FIT"))
    client.get_activity_streams = AsyncMock(return_value={})
    return MagicMock(return_value=client)


@pytest.fixture
async def user_db(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'user.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(UserBase.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        session.add(Athlete(id="athlete-1", global_user_id=_USER_ID, ftp_tests=[]))
        await session.commit()
    yield factory
    await engine.dispose()


async def _run_sync(factory) -> int:
    async with factory() as session:
        athlete = (
            await session.execute(select(Athlete).where(Athlete.id == "athlete-1"))
        ).scalar_one()
        conn = MagicMock(spec=ProviderConnection)
        conn.user_id = _USER_ID
        conn.provider = "strava"
        with patch(
            "backend.app.services.provider_sync.PROVIDERS", {"strava": _provider()}
        ):
            created, _ = await sync_provider_activities(
                athlete, conn, session, user_id=_USER_ID, access_token="tok"
            )
        return created


class TestUsageFailureIsInvisibleToTheSync:
    async def test_a_locked_usage_db_still_imports_the_activity(
        self, user_db, api_usage_db, monkeypatch
    ):
        def locked():
            raise RuntimeError("database is locked")

        monkeypatch.setattr(api_usage_service, "api_usage_session_factory", locked)

        assert await _run_sync(user_db) == 1

        async with user_db() as session:
            activities = (await session.execute(select(Activity))).scalars().all()
        assert len(activities) == 1, "the import survived the accounting failure"

    async def test_a_working_usage_db_does_record(self, api_usage_db):
        """The other half of the contract: when it *does* work, it works.

        A file that only forced the failure would pass just as happily against
        accounting that never recorded anything at all.
        """
        from backend.app.models.api_usage_orm import ApiUsage

        with api_usage_service.attribute_to_user(_USER_ID):
            await api_usage_service.record_api_usage(
                service="strava",
                endpoint="/athlete/activities",
                method="GET",
                status_code=200,
            )

        async with api_usage_db() as session:
            (row,) = (await session.execute(select(ApiUsage))).scalars().all()
        assert row.user_id == _USER_ID
        assert row.outcome == "ok"

    async def test_the_sync_runs_inside_an_attribution_scope(self, user_db, api_usage_db):
        """`sync_provider_activities` establishes the athlete for its whole walk."""
        seen: list[str | None] = []

        async def capturing_list_activities(*args, **kwargs):
            seen.append(api_usage_service.current_user_id())
            return []

        client = MagicMock()
        client.list_activities = AsyncMock(side_effect=capturing_list_activities)

        async with user_db() as session:
            athlete = (
                await session.execute(select(Athlete).where(Athlete.id == "athlete-1"))
            ).scalar_one()
            conn = MagicMock(spec=ProviderConnection)
            conn.user_id = _USER_ID
            conn.provider = "strava"
            with patch(
                "backend.app.services.provider_sync.PROVIDERS",
                {"strava": MagicMock(return_value=client)},
            ):
                await sync_provider_activities(
                    athlete, conn, session, user_id=_USER_ID, access_token="tok"
                )

        assert seen == [_USER_ID]
        # And the scope does not leak past the sync.
        assert api_usage_service.current_user_id() is None
