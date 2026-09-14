"""
Unit tests for backend.app.services.provider_sync.

Tests ensure_fresh_token and sync_provider_activities in isolation by mocking
the PROVIDERS registry so no real HTTP calls are made.
"""
import asyncio
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.db.base import UserBase
from backend.app.models.user_orm import Activity, ActivitySource, Athlete
from backend.app.models.registry_orm import ProviderConnection
from backend.app.services.provider_sync import ensure_fresh_token, sync_provider_activities
from backend.app.services.providers.base import NormalizedActivity


# ── Helpers ────────────────────────────────────────────────────────────────────


_TEST_USER_ID = "test-user-00000000"  # the user conftest's registry_session seeds


async def _real_conn(
    registry_session,
    provider: str = "strava",
    *,
    access_token: str = "access-tok",
    refresh_token: str = "refresh-tok",
    token_expires_at: datetime | None = None,
) -> ProviderConnection:
    """A persisted ProviderConnection row.

    A mock will not do here any more: ``ensure_fresh_token`` claims the right to
    rotate with a conditional UPDATE against this row (issue #50), so the row has
    to exist for the claim to be winnable.
    """
    conn = ProviderConnection(
        user_id=_TEST_USER_ID,
        provider=provider,
        access_token=access_token,
        refresh_token=refresh_token,
        token_expires_at=token_expires_at,
    )
    registry_session.add(conn)
    await registry_session.commit()
    await registry_session.refresh(conn)
    return conn


def _norm(
    ext_id: str = "act-1",
    source: str = "strava",
    start_time: datetime | None = None,
) -> NormalizedActivity:
    return NormalizedActivity(
        external_id=ext_id,
        source=source,
        name="Test Ride",
        sport_type="Ride",
        start_time=start_time or datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
        duration_s=3600,
        distance_m=50_000.0,
        elevation_m=500.0,
        avg_power=None,
        avg_hr=None,
        max_hr=None,
        avg_speed_ms=14.0,
        avg_cadence=None,
    )


async def _make_athlete(session, user_id: str = "user-1") -> Athlete:
    athlete = Athlete(global_user_id=user_id, ftp_tests=[])
    session.add(athlete)
    await session.commit()
    await session.refresh(athlete)
    return athlete


def _make_connection(
    athlete: Athlete,
    provider: str = "strava",
    expires_in: timedelta = timedelta(hours=1),
) -> ProviderConnection:
    conn = MagicMock(spec=ProviderConnection)
    conn.user_id = athlete.global_user_id
    conn.provider = provider
    conn.access_token = "access-tok"
    conn.refresh_token = "refresh-tok"
    conn.token_expires_at = datetime.now(timezone.utc) + expires_in
    return conn


_TEAM_ID = "test-team"
_ACCESS_TOKEN = "access-tok"


# ── ensure_fresh_token ─────────────────────────────────────────────────────────


class TestEnsureFreshToken:
    async def test_valid_token_returned_unchanged(self, registry_session):
        conn = await _real_conn(
            registry_session,
            token_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        token = await ensure_fresh_token(conn, registry_session)
        assert token == "access-tok"

    async def test_no_expiry_returns_current_token(self, registry_session):
        conn = await _real_conn(registry_session, token_expires_at=None)
        token = await ensure_fresh_token(conn, registry_session)
        assert token == "access-tok"

    async def test_expired_token_is_refreshed(self, registry_session):
        conn = await _real_conn(
            registry_session,
            token_expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        mock_cls = MagicMock()
        mock_cls.refresh_access_token = AsyncMock(
            return_value={
                "access_token": "refreshed-token",
                "refresh_token": "new-refresh",
                "expires_at": 9999999999,
            }
        )

        with patch("backend.app.services.provider_sync.PROVIDERS", {"strava": mock_cls}):
            token = await ensure_fresh_token(conn, registry_session)

        assert token == "refreshed-token"
        mock_cls.refresh_access_token.assert_called_once_with("refresh-tok")

    async def test_expired_token_updates_connection_attributes(self, registry_session):
        conn = await _real_conn(
            registry_session,
            token_expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        mock_cls = MagicMock()
        mock_cls.refresh_access_token = AsyncMock(
            return_value={
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_at": 9999999999,
            }
        )

        with patch("backend.app.services.provider_sync.PROVIDERS", {"strava": mock_cls}):
            await ensure_fresh_token(conn, registry_session)

        assert conn.access_token == "new-access"
        assert conn.refresh_token == "new-refresh"

    async def test_unknown_provider_returns_current_token_without_error(
        self, registry_session
    ):
        conn = await _real_conn(
            registry_session,
            provider="nonexistent",
            token_expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        with patch("backend.app.services.provider_sync.PROVIDERS", {}):
            token = await ensure_fresh_token(conn, registry_session)
        assert token == "access-tok"

    async def test_strava_token_expiring_within_30_minutes_is_refreshed(
        self, registry_session
    ):
        conn = await _real_conn(
            registry_session,
            provider="strava",
            token_expires_at=datetime.now(timezone.utc) + timedelta(minutes=20),
        )
        mock_cls = MagicMock()
        mock_cls.refresh_access_token = AsyncMock(
            return_value={
                "access_token": "proactive-strava",
                "refresh_token": "new-refresh",
                "expires_at": 9999999999,
            }
        )
        with patch("backend.app.services.provider_sync.PROVIDERS", {"strava": mock_cls}):
            token = await ensure_fresh_token(conn, registry_session)
        assert token == "proactive-strava"

    async def test_strava_token_with_over_30_minutes_is_not_refreshed(
        self, registry_session
    ):
        conn = await _real_conn(
            registry_session,
            provider="strava",
            token_expires_at=datetime.now(timezone.utc) + timedelta(minutes=45),
        )
        mock_cls = MagicMock()
        mock_cls.refresh_access_token = AsyncMock()
        with patch("backend.app.services.provider_sync.PROVIDERS", {"strava": mock_cls}):
            token = await ensure_fresh_token(conn, registry_session)
        assert token == "access-tok"
        mock_cls.refresh_access_token.assert_not_called()

    async def test_wahoo_token_expiring_within_1_minute_is_refreshed(
        self, registry_session
    ):
        conn = await _real_conn(
            registry_session,
            provider="wahoo",
            token_expires_at=datetime.now(timezone.utc) + timedelta(seconds=30),
        )
        mock_cls = MagicMock()
        mock_cls.refresh_access_token = AsyncMock(
            return_value={
                "access_token": "proactive-wahoo",
                "refresh_token": "new-refresh",
                "expires_at": 9999999999,
            }
        )
        with patch("backend.app.services.provider_sync.PROVIDERS", {"wahoo": mock_cls}):
            token = await ensure_fresh_token(conn, registry_session)
        assert token == "proactive-wahoo"

    async def test_wahoo_token_with_over_1_minute_is_not_refreshed(
        self, registry_session
    ):
        conn = await _real_conn(
            registry_session,
            provider="wahoo",
            token_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        mock_cls = MagicMock()
        mock_cls.refresh_access_token = AsyncMock()
        with patch("backend.app.services.provider_sync.PROVIDERS", {"wahoo": mock_cls}):
            token = await ensure_fresh_token(conn, registry_session)
        assert token == "access-tok"
        mock_cls.refresh_access_token.assert_not_called()

    async def test_refresh_failure_is_logged_and_reraised(self, registry_session):
        conn = await _real_conn(
            registry_session,
            token_expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        mock_cls = MagicMock()
        mock_cls.refresh_access_token = AsyncMock(side_effect=RuntimeError("provider down"))

        with (
            patch("backend.app.services.provider_sync.PROVIDERS", {"strava": mock_cls}),
            patch("backend.app.services.provider_sync.log") as mock_log,
        ):
            with pytest.raises(RuntimeError, match="provider down"):
                await ensure_fresh_token(conn, registry_session)

        mock_log.error.assert_called_once()
        call_args = mock_log.error.call_args
        assert "strava" in call_args.args[1]

    async def test_a_failed_refresh_releases_the_claim(self, registry_session):
        """The next caller must get to try, not wait out the whole TTL."""
        conn = await _real_conn(
            registry_session,
            token_expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        failing = MagicMock()
        failing.refresh_access_token = AsyncMock(side_effect=RuntimeError("provider down"))

        with patch("backend.app.services.provider_sync.PROVIDERS", {"strava": failing}):
            with pytest.raises(RuntimeError):
                await ensure_fresh_token(conn, registry_session)

        await registry_session.refresh(conn)
        assert conn.refresh_lock_until is None

        # And a second attempt gets as far as the provider rather than waiting.
        recovering = MagicMock()
        recovering.refresh_access_token = AsyncMock(
            return_value={
                "access_token": "second-try",
                "refresh_token": "new-refresh",
                "expires_at": 9999999999,
            }
        )
        with patch("backend.app.services.provider_sync.PROVIDERS", {"strava": recovering}):
            assert await ensure_fresh_token(conn, registry_session) == "second-try"

    async def test_a_successful_refresh_releases_the_claim(self, registry_session):
        conn = await _real_conn(
            registry_session,
            token_expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        mock_cls = MagicMock()
        mock_cls.refresh_access_token = AsyncMock(
            return_value={
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_at": 9999999999,
            }
        )
        with patch("backend.app.services.provider_sync.PROVIDERS", {"strava": mock_cls}):
            await ensure_fresh_token(conn, registry_session)

        await registry_session.refresh(conn)
        assert conn.refresh_lock_until is None


# ── sync_provider_activities ───────────────────────────────────────────────────


class TestSyncProviderActivities:
    async def test_imports_new_activity_creates_source(self, session):
        """A new activity creates exactly one Activity + one ActivitySource."""
        athlete = await _make_athlete(session)
        conn = _make_connection(athlete)

        mock_client = MagicMock()
        mock_client.list_activities = AsyncMock(side_effect=[[_norm()], []])
        mock_client.download_fit_file = AsyncMock(side_effect=Exception("no FIT"))
        mock_client.get_activity_streams = AsyncMock(return_value={})
        mock_cls = MagicMock(return_value=mock_client)

        with patch("backend.app.services.provider_sync.PROVIDERS", {"strava": mock_cls}):
            count, earliest = await sync_provider_activities(
                athlete, conn, session, user_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        assert count == 1
        assert earliest == date(2024, 6, 1)

        # Verify Activity + ActivitySource were created
        acts = (await session.execute(select(Activity).where(Activity.athlete_id == athlete.id))).scalars().all()
        assert len(acts) == 1
        srcs = (await session.execute(select(ActivitySource).where(ActivitySource.activity_id == acts[0].id))).scalars().all()
        assert len(srcs) == 1
        assert srcs[0].provider == "strava"
        assert srcs[0].external_id == "act-1"

    async def test_skips_already_imported_source(self, session):
        """If (provider, external_id) already has an ActivitySource, skip it."""
        athlete = await _make_athlete(session, user_id="user-2")
        conn = _make_connection(athlete)

        # Pre-seed Activity + ActivitySource. `streams_fetched_at` is what makes
        # this a *finished* import rather than one that fell short: a source
        # without it is repaired instead of skipped (issue #67), which is the
        # case `TestRepairingIncompleteImports` covers.
        act = Activity(
            athlete_id=athlete.id,
            start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
            duration_s=3600,
            status="processed",
        )
        session.add(act)
        await session.flush()
        session.add(
            ActivitySource(
                activity_id=act.id,
                provider="strava",
                external_id="act-1",
                streams_fetched_at=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
            )
        )
        await session.commit()

        mock_client = MagicMock()
        mock_client.list_activities = AsyncMock(side_effect=[[_norm()], []])
        mock_client.download_fit_file = AsyncMock(side_effect=Exception("no FIT"))
        mock_client.get_activity_streams = AsyncMock(return_value={})
        mock_cls = MagicMock(return_value=mock_client)

        with patch("backend.app.services.provider_sync.PROVIDERS", {"strava": mock_cls}):
            count, earliest = await sync_provider_activities(
                athlete, conn, session, user_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        assert count == 0
        assert earliest is None

    async def test_same_workout_second_provider_adds_source_to_existing_activity(self, session):
        """When a second provider syncs the same workout, it adds an ActivitySource
        to the existing Activity instead of creating a new one."""
        athlete = await _make_athlete(session, user_id="user-3")
        strava_conn = _make_connection(athlete, provider="strava")
        wahoo_conn = _make_connection(athlete, provider="wahoo")

        base_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)

        # Sync Strava first
        strava_mock = MagicMock()
        strava_mock.list_activities = AsyncMock(side_effect=[[_norm("strava-1", "strava", base_time)], []])
        strava_mock.download_fit_file = AsyncMock(side_effect=Exception("no FIT"))
        strava_mock.get_activity_streams = AsyncMock(return_value={})
        strava_cls = MagicMock(return_value=strava_mock)

        with patch("backend.app.services.provider_sync.PROVIDERS", {"strava": strava_cls}):
            await sync_provider_activities(
                athlete, strava_conn, session, user_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        # Sync Wahoo — same start_time, should attach to existing Activity
        wahoo_mock = MagicMock()
        wahoo_mock.list_activities = AsyncMock(side_effect=[[_norm("wahoo-1", "wahoo", base_time)], []])
        wahoo_mock.download_fit_file = AsyncMock(side_effect=Exception("no FIT"))
        wahoo_mock.get_activity_streams = AsyncMock(return_value={})
        wahoo_cls = MagicMock(return_value=wahoo_mock)

        with patch("backend.app.services.provider_sync.PROVIDERS", {"wahoo": wahoo_cls}):
            await sync_provider_activities(
                athlete, wahoo_conn, session, user_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        # Exactly ONE Activity, TWO ActivitySources
        acts = (await session.execute(select(Activity).where(Activity.athlete_id == athlete.id))).scalars().all()
        assert len(acts) == 1

        srcs = (await session.execute(select(ActivitySource).where(ActivitySource.activity_id == acts[0].id))).scalars().all()
        providers = {s.provider for s in srcs}
        assert providers == {"strava", "wahoo"}

    async def test_wahoo_with_fit_repopulates_when_strava_is_existing_winner(self, session):
        """Wahoo with a FIT file (priority=2) beats an existing Strava source (priority=3)
        and repopulates the Activity metrics."""
        athlete = await _make_athlete(session, user_id="user-4")
        athlete.ftp = 250
        await session.commit()

        strava_conn = _make_connection(athlete, provider="strava")
        wahoo_conn = _make_connection(athlete, provider="wahoo")

        base_time = datetime(2024, 7, 1, 8, 0, tzinfo=timezone.utc)

        # Strava syncs first with power stream data
        strava_mock = MagicMock()
        strava_mock.list_activities = AsyncMock(side_effect=[[_norm("strava-1", "strava", base_time)], []])
        strava_mock.download_fit_file = AsyncMock(side_effect=Exception("no FIT"))
        strava_mock.get_activity_streams = AsyncMock(return_value={"power": [150] * 120})
        strava_cls = MagicMock(return_value=strava_mock)

        with patch("backend.app.services.provider_sync.PROVIDERS", {"strava": strava_cls}):
            await sync_provider_activities(
                athlete, strava_conn, session, user_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        # Capture Strava-derived Load
        acts = (await session.execute(select(Activity).where(Activity.athlete_id == athlete.id))).scalars().all()
        assert len(acts) == 1
        strava_tss = acts[0].load

        # Wahoo syncs with a FIT file — should repopulate (priority 2 beats priority 3)
        fit_bytes = b"fakeFITdata"
        wahoo_mock = MagicMock()
        wahoo_mock.list_activities = AsyncMock(side_effect=[[_norm("wahoo-1", "wahoo", base_time)], []])
        # Return fake FIT bytes so we know FIT was "downloaded"
        wahoo_mock.download_fit_file = AsyncMock(return_value=fit_bytes)
        wahoo_cls = MagicMock(return_value=wahoo_mock)

        from unittest.mock import patch as _patch
        import fitdecode

        fake_profile = MagicMock()
        fake_profile.power = [200] * 120
        fake_profile.heartRate = []
        fake_profile.cadence = []
        fake_profile.speed = []
        fake_profile.altitude = []
        fake_profile.avgHeartRate = None
        fake_profile.peakHR = None
        fake_profile.avgPower = 200.0
        fake_profile.avgCadence = 0
        fake_profile.avgSpeed = 0
        fake_profile.duration = 3600
        fake_profile.distance = 50000
        fake_profile.elevationGain = 500
        fake_profile.start_time = base_time
        fake_profile.sport_type = "cycling"

        with (
            patch("backend.app.services.provider_sync.PROVIDERS", {"wahoo": wahoo_cls}),
            patch("backend.app.services.provider_sync.summarizeWorkout", return_value=fake_profile),
            patch("backend.app.services.provider_sync.encrypt_file"),
        ):
            wahoo_count, _ = await sync_provider_activities(
                athlete, wahoo_conn, session, user_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        assert wahoo_count == 1

        await session.refresh(acts[0])
        # Activity should now have Wahoo FIT data (higher power → higher Load)
        assert acts[0].load is not None
        # Two sources on the single Activity
        srcs = (await session.execute(select(ActivitySource).where(ActivitySource.activity_id == acts[0].id))).scalars().all()
        assert {s.provider for s in srcs} == {"strava", "wahoo"}

    async def test_lower_priority_source_does_not_repopulate(self, session):
        """Strava (priority=3) does not repopulate when Wahoo+FIT (priority=2) is existing winner."""
        athlete = await _make_athlete(session, user_id="user-5")
        athlete.ftp = 250
        await session.commit()

        wahoo_conn = _make_connection(athlete, provider="wahoo")
        strava_conn = _make_connection(athlete, provider="strava")

        base_time = datetime(2024, 7, 2, 8, 0, tzinfo=timezone.utc)

        # Wahoo syncs first with a FIT file (priority=2)
        fit_bytes = b"fakeFIT"
        wahoo_mock = MagicMock()
        wahoo_mock.list_activities = AsyncMock(side_effect=[[_norm("wahoo-1", "wahoo", base_time)], []])
        wahoo_mock.download_fit_file = AsyncMock(return_value=fit_bytes)
        wahoo_cls = MagicMock(return_value=wahoo_mock)

        fake_profile = MagicMock()
        fake_profile.power = [220] * 3600
        fake_profile.heartRate = []
        fake_profile.cadence = []
        fake_profile.speed = []
        fake_profile.altitude = []
        fake_profile.avgHeartRate = None
        fake_profile.peakHR = None
        fake_profile.avgPower = 220.0
        fake_profile.avgCadence = 0
        fake_profile.avgSpeed = 0
        fake_profile.duration = 3600
        fake_profile.distance = 50000
        fake_profile.elevationGain = 500
        fake_profile.start_time = base_time
        fake_profile.sport_type = "cycling"

        with (
            patch("backend.app.services.provider_sync.PROVIDERS", {"wahoo": wahoo_cls}),
            patch("backend.app.services.provider_sync.summarizeWorkout", return_value=fake_profile),
            patch("backend.app.services.provider_sync.encrypt_file"),
        ):
            await sync_provider_activities(
                athlete, wahoo_conn, session, user_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        acts = (await session.execute(select(Activity).where(Activity.athlete_id == athlete.id))).scalars().all()
        assert len(acts) == 1
        wahoo_tss = acts[0].load

        # Strava syncs with different stream data — should NOT repopulate (priority 3 > 2)
        strava_mock = MagicMock()
        strava_mock.list_activities = AsyncMock(side_effect=[[_norm("strava-1", "strava", base_time)], []])
        strava_mock.download_fit_file = AsyncMock(side_effect=Exception("no FIT"))
        strava_mock.get_activity_streams = AsyncMock(return_value={"power": [100] * 120})
        strava_cls = MagicMock(return_value=strava_mock)

        with patch("backend.app.services.provider_sync.PROVIDERS", {"strava": strava_cls}):
            strava_count, _ = await sync_provider_activities(
                athlete, strava_conn, session, user_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        assert strava_count == 0  # Strava source added but not counted as a new/updated activity

        await session.refresh(acts[0])
        # Metrics should be unchanged from Wahoo's data
        assert acts[0].load == wahoo_tss

        srcs = (await session.execute(select(ActivitySource).where(ActivitySource.activity_id == acts[0].id))).scalars().all()
        assert {s.provider for s in srcs} == {"wahoo", "strava"}

    async def test_blank_wahoo_strava_with_data_becomes_winner(self, session):
        """Wahoo without FIT (priority=4) is already existing; Strava (priority=3) wins
        and repopulates the Activity metrics."""
        athlete = await _make_athlete(session, user_id="user-6")
        athlete.ftp = 250
        await session.commit()

        wahoo_conn = _make_connection(athlete, provider="wahoo")
        strava_conn = _make_connection(athlete, provider="strava")

        base_time = datetime(2024, 7, 5, 8, 0, tzinfo=timezone.utc)

        # Wahoo syncs first — no FIT, no streams → blank (priority=4)
        wahoo_mock = MagicMock()
        wahoo_mock.list_activities = AsyncMock(side_effect=[[_norm("wahoo-blank", "wahoo", base_time)], []])
        wahoo_mock.download_fit_file = AsyncMock(side_effect=Exception("no FIT"))
        wahoo_mock.get_activity_streams = AsyncMock(return_value={})
        wahoo_cls = MagicMock(return_value=wahoo_mock)

        with patch("backend.app.services.provider_sync.PROVIDERS", {"wahoo": wahoo_cls}):
            await sync_provider_activities(
                athlete, wahoo_conn, session, user_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        acts = (await session.execute(select(Activity).where(Activity.athlete_id == athlete.id))).scalars().all()
        assert len(acts) == 1
        assert acts[0].load is None  # blank Wahoo has no Load

        # Strava syncs with power data → priority=3 beats blank Wahoo priority=4 → repopulates
        strava_mock = MagicMock()
        strava_mock.list_activities = AsyncMock(side_effect=[[_norm("strava-real", "strava", base_time)], []])
        strava_mock.download_fit_file = AsyncMock(side_effect=Exception("no FIT"))
        strava_mock.get_activity_streams = AsyncMock(return_value={"power": [200] * 120})
        strava_cls = MagicMock(return_value=strava_mock)

        with patch("backend.app.services.provider_sync.PROVIDERS", {"strava": strava_cls}):
            strava_count, _ = await sync_provider_activities(
                athlete, strava_conn, session, user_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        assert strava_count == 1  # repopulated

        await session.refresh(acts[0])
        # Activity should now have Strava's data
        assert acts[0].load is not None

        srcs = (await session.execute(select(ActivitySource).where(ActivitySource.activity_id == acts[0].id))).scalars().all()
        assert {s.provider for s in srcs} == {"wahoo", "strava"}

    async def test_returns_correct_count_and_earliest_date(self, session):
        athlete = await _make_athlete(session, user_id="user-7")
        conn = _make_connection(athlete)

        activities = [
            _norm(
                ext_id=f"act-{i}",
                start_time=datetime(2024, 6, i, 10, 0, tzinfo=timezone.utc),
            )
            for i in range(1, 4)  # June 1, 2, 3
        ]
        mock_client = MagicMock()
        mock_client.list_activities = AsyncMock(side_effect=[activities, []])
        mock_client.download_fit_file = AsyncMock(side_effect=Exception("no FIT"))
        mock_client.get_activity_streams = AsyncMock(return_value={})
        mock_cls = MagicMock(return_value=mock_client)

        with patch("backend.app.services.provider_sync.PROVIDERS", {"strava": mock_cls}):
            count, earliest = await sync_provider_activities(
                athlete, conn, session, user_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        assert count == 3
        assert earliest == date(2024, 6, 1)

    async def test_stream_data_persisted_with_activity(self, session):
        from backend.app.models.user_orm import ActivityStream

        athlete = await _make_athlete(session, user_id="user-8")
        conn = _make_connection(athlete)

        mock_client = MagicMock()
        mock_client.list_activities = AsyncMock(side_effect=[[_norm()], []])
        mock_client.download_fit_file = AsyncMock(side_effect=Exception("no FIT"))
        mock_client.get_activity_streams = AsyncMock(
            return_value={"power": [200, 210, 220], "heartrate": [140, 145, 150]}
        )
        mock_cls = MagicMock(return_value=mock_client)

        with patch("backend.app.services.provider_sync.PROVIDERS", {"strava": mock_cls}):
            count, _ = await sync_provider_activities(
                athlete, conn, session, user_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        assert count == 1

        act = (await session.execute(
            select(Activity).where(Activity.athlete_id == athlete.id)
        )).scalar_one()
        stream_result = await session.execute(
            select(ActivityStream).where(ActivityStream.activity_id == act.id)
        )
        stream_types = {s.stream_type for s in stream_result.scalars()}
        assert "power" in stream_types
        assert "heartrate" in stream_types

    async def test_unknown_provider_returns_zero(self, session):
        athlete = await _make_athlete(session, user_id="user-9")
        conn = _make_connection(athlete, provider="unknown")

        with patch("backend.app.services.provider_sync.PROVIDERS", {}):
            count, earliest = await sync_provider_activities(
                athlete, conn, session, user_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        assert count == 0
        assert earliest is None

    async def test_pagination_stops_on_empty_page(self, session):
        """list_activities is called until it returns an empty list."""
        athlete = await _make_athlete(session, user_id="user-10")
        conn = _make_connection(athlete)

        # Each activity has a distinct start_time so they aren't merged
        t1 = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
        t2 = datetime(2024, 6, 2, 10, 0, tzinfo=timezone.utc)
        t3 = datetime(2024, 6, 3, 10, 0, tzinfo=timezone.utc)

        mock_client = MagicMock()
        mock_client.list_activities = AsyncMock(
            side_effect=[
                [_norm("a1", start_time=t1), _norm("a2", start_time=t2)],
                [_norm("a3", start_time=t3)],
                [],
            ]
        )
        mock_client.download_fit_file = AsyncMock(side_effect=Exception("no FIT"))
        mock_client.get_activity_streams = AsyncMock(return_value={})
        mock_cls = MagicMock(return_value=mock_client)

        with patch("backend.app.services.provider_sync.PROVIDERS", {"strava": mock_cls}):
            count, _ = await sync_provider_activities(
                athlete, conn, session, user_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        assert count == 3
        assert mock_client.list_activities.call_count == 3

    async def test_auto_category_assigned_from_stream_data(self, session):
        """workout_category is auto-assigned when power streams are available."""
        athlete = await _make_athlete(session, user_id="user-11")
        athlete.ftp = 250
        await session.commit()

        conn = _make_connection(athlete)

        mock_client = MagicMock()
        mock_client.list_activities = AsyncMock(side_effect=[[_norm()], []])
        mock_client.download_fit_file = AsyncMock(side_effect=Exception("no FIT"))
        # 3600 samples at 200 W → Intensity = 200/250 = 0.80 → "tempo"
        mock_client.get_activity_streams = AsyncMock(
            return_value={"power": [200] * 3600}
        )
        mock_cls = MagicMock(return_value=mock_client)

        with patch("backend.app.services.provider_sync.PROVIDERS", {"strava": mock_cls}):
            await sync_provider_activities(
                athlete, conn, session, user_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        act = (await session.execute(
            select(Activity).where(Activity.athlete_id == athlete.id)
        )).scalar_one()
        assert act.workout_category == "tempo"

    async def test_auto_category_assigned_from_fit_file(self, session):
        """workout_category is auto-assigned when a FIT file is processed during sync."""
        athlete = await _make_athlete(session, user_id="user-12")
        athlete.ftp = 250
        await session.commit()

        conn = _make_connection(athlete, provider="wahoo")
        base_time = datetime(2024, 8, 1, 9, 0, tzinfo=timezone.utc)

        fit_bytes = b"fakeFITdata"
        wahoo_mock = MagicMock()
        wahoo_mock.list_activities = AsyncMock(side_effect=[[_norm("wahoo-1", "wahoo", base_time)], []])
        wahoo_mock.download_fit_file = AsyncMock(return_value=fit_bytes)
        wahoo_cls = MagicMock(return_value=wahoo_mock)

        fake_profile = MagicMock()
        # 3600 samples at 225 W → Intensity = 225/250 = 0.90 → "threshold"
        fake_profile.power = [225] * 3600
        fake_profile.heartRate = []
        fake_profile.cadence = []
        fake_profile.speed = []
        fake_profile.altitude = []
        fake_profile.avgHeartRate = None
        fake_profile.peakHR = None
        fake_profile.avgPower = 225.0
        fake_profile.avgCadence = 0
        fake_profile.avgSpeed = 0
        fake_profile.duration = 3600
        fake_profile.distance = 50000
        fake_profile.elevationGain = 500
        fake_profile.start_time = base_time
        fake_profile.sport_type = "cycling"

        with (
            patch("backend.app.services.provider_sync.PROVIDERS", {"wahoo": wahoo_cls}),
            patch("backend.app.services.provider_sync.summarizeWorkout", return_value=fake_profile),
            patch("backend.app.services.provider_sync.encrypt_file"),
        ):
            await sync_provider_activities(
                athlete, conn, session, user_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        act = (await session.execute(
            select(Activity).where(Activity.athlete_id == athlete.id)
        )).scalar_one()
        assert act.workout_category == "threshold"

    async def test_no_category_when_no_power_data(self, session):
        """workout_category stays None when there is no power data to classify from."""
        athlete = await _make_athlete(session, user_id="user-13")
        athlete.ftp = 250
        await session.commit()

        conn = _make_connection(athlete)

        mock_client = MagicMock()
        mock_client.list_activities = AsyncMock(side_effect=[[_norm()], []])
        mock_client.download_fit_file = AsyncMock(side_effect=Exception("no FIT"))
        mock_client.get_activity_streams = AsyncMock(return_value={})  # no power
        mock_cls = MagicMock(return_value=mock_client)

        with patch("backend.app.services.provider_sync.PROVIDERS", {"strava": mock_cls}):
            await sync_provider_activities(
                athlete, conn, session, user_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        act = (await session.execute(
            select(Activity).where(Activity.athlete_id == athlete.id)
        )).scalar_one()
        assert act.workout_category is None


# ── Dedup race condition (issue #76) ──────────────────────────────────────────


class TestDeduplicationRaceCondition:
    """
    Tests for the cross-session dedup race condition reported in #76.

    Root cause
    ----------
    The asyncio lock in provider_sync serialises the dedup-window query + flush
    *within a single process*, but it used to release before the database
    transaction committed.  Under READ COMMITTED isolation (and SQLite WAL mode),
    a second session that acquired the lock after the first had flushed-but-not-
    committed would see an empty dedup window and create a duplicate Activity.

    Fix
    ---
    ``await session.commit()`` is now called inside the asyncio lock for the
    "new workout" path in provider_sync.py, wahoo_sync.py, and strava_sync.py.
    The commit makes the new Activity visible to other sessions before the lock
    is released, closing the window entirely.

    Why a file-based engine is required here
    ----------------------------------------
    ``sqlite+aiosqlite:///:memory:`` gives every aiosqlite connection its own
    private in-memory database, so separate sessions literally cannot share data.
    A file-based database allows multiple connections to the same database, which
    is what we need to observe the cross-session isolation behaviour.
    SQLite's single-writer semantics still apply, but the observable outcome
    faithfully reproduces the PostgreSQL race.
    """

    @pytest.fixture
    async def race_engine(self, tmp_path):
        """File-based SQLite engine shared by multiple concurrent sessions."""
        db_path = tmp_path / "race.db"
        eng = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            # Allow a short busy-wait so the second writer can proceed once the
            # first session commits rather than failing immediately.
            connect_args={"timeout": 10},
        )
        async with eng.begin() as conn:
            await conn.run_sync(UserBase.metadata.create_all)
        yield eng
        await eng.dispose()

    async def test_flush_without_commit_allows_race(self, race_engine):
        """
        Documents the isolation problem: releasing the lock after a flush (but
        before the commit) allows a second concurrent session to see an empty
        dedup window and create a duplicate Activity.

        This test intentionally uses the OLD buggy pattern and asserts that two
        activities are created, proving the race mechanism is real.  It should
        always pass regardless of production-code changes.

        Two asyncio.Event objects gate the ordering deterministically:
        - first_released_lock: first_task signals it after releasing the asyncio
          lock (still holding the SQLite RESERVED lock, before commit).
        - second_has_read: second_task signals it after executing the dedup SELECT
          but BEFORE its own flush.  Signalling before flush is critical: the flush
          will block in the aiosqlite background thread (INSERT waits for the
          RESERVED lock held by first_task).  Signalling first lets first_task
          proceed to commit, releasing the RESERVED lock and unblocking the INSERT.
        """
        from backend.app.services.provider_sync import _DUPLICATE_WINDOW, _get_activity_lock

        factory = async_sessionmaker(race_engine, expire_on_commit=False)
        base_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)

        async with factory() as s:
            athlete = Athlete(global_user_id="race-flush-user", ftp_tests=[])
            s.add(athlete)
            await s.commit()
            athlete_id = athlete.id

        team_id = f"race-flush-{athlete_id}"
        first_released_lock = asyncio.Event()
        second_has_read = asyncio.Event()

        async def first_task() -> None:
            async with factory() as s:
                async with _get_activity_lock(team_id, athlete_id):
                    r = await s.execute(
                        select(Activity).where(
                            Activity.athlete_id == athlete_id,
                            Activity.start_time >= base_time - _DUPLICATE_WINDOW,
                            Activity.start_time <= base_time + _DUPLICATE_WINDOW,
                        )
                    )
                    if r.scalar_one_or_none() is None:
                        s.add(Activity(
                            athlete_id=athlete_id,
                            start_time=base_time,
                            duration_s=5010,
                            status="pending",
                        ))
                        await s.flush()
                # Lock released WITHOUT committing (the bug).
                # Signal second_task to acquire the lock and read.
                first_released_lock.set()
                # Wait until second_task has read so its SELECT is guaranteed to
                # land while first_task's RESERVED lock is still held (uncommitted).
                await second_has_read.wait()
                await s.commit()

        async def second_task() -> None:
            # Wait until first_task has released the asyncio lock before trying
            # to acquire it, so ordering is guaranteed.
            await first_released_lock.wait()
            async with factory() as s:
                async with _get_activity_lock(team_id, athlete_id):
                    r = await s.execute(
                        select(Activity).where(
                            Activity.athlete_id == athlete_id,
                            Activity.start_time >= base_time - _DUPLICATE_WINDOW,
                            Activity.start_time <= base_time + _DUPLICATE_WINDOW,
                        )
                    )
                    found = r.scalar_one_or_none()
                    # Signal BEFORE flush: the flush blocks in the aiosqlite
                    # background thread (INSERT waits for first_task's RESERVED
                    # lock).  Setting the event here lets the event loop run
                    # first_task → commit → release RESERVED → unblock INSERT.
                    second_has_read.set()
                    if found is None:
                        s.add(Activity(
                            athlete_id=athlete_id,
                            start_time=base_time,
                            duration_s=5009,
                            status="pending",
                        ))
                        await s.flush()
                await s.commit()

        await asyncio.gather(first_task(), second_task())

        async with factory() as s:
            r = await s.execute(select(Activity).where(Activity.athlete_id == athlete_id))
            activities = r.scalars().all()

        # Demonstrates the race: flush-without-commit leaves a window that the
        # second session exploits, producing a duplicate row.
        assert len(activities) == 2, (
            f"Expected 2 (race condition produces a duplicate) but got {len(activities)}. "
            "The asyncio.Event gates should make the ordering deterministic."
        )

    async def test_commit_inside_lock_prevents_race(self, race_engine):
        """
        Regression test for #76: committing inside the asyncio lock ensures that
        the new Activity is visible to any concurrent session before the lock is
        released, so the second session finds the existing Activity and does not
        create a duplicate.
        """
        from backend.app.services.provider_sync import _DUPLICATE_WINDOW, _get_activity_lock

        factory = async_sessionmaker(race_engine, expire_on_commit=False)
        base_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)

        async with factory() as s:
            athlete = Athlete(global_user_id="race-fix-user", ftp_tests=[])
            s.add(athlete)
            await s.commit()
            athlete_id = athlete.id

        team_id = f"race-fix-{athlete_id}"

        async def first_task() -> None:
            async with factory() as s:
                async with _get_activity_lock(team_id, athlete_id):
                    r = await s.execute(
                        select(Activity).where(
                            Activity.athlete_id == athlete_id,
                            Activity.start_time >= base_time - _DUPLICATE_WINDOW,
                            Activity.start_time <= base_time + _DUPLICATE_WINDOW,
                        )
                    )
                    if r.scalar_one_or_none() is None:
                        s.add(Activity(
                            athlete_id=athlete_id,
                            start_time=base_time,
                            duration_s=5010,
                            status="pending",
                        ))
                        await s.flush()
                    # Fix: commit inside the lock so data is visible before
                    # any other session acquires the lock.
                    await s.commit()

        async def second_task() -> None:
            await asyncio.sleep(0)  # yield so first_task acquires the lock first
            async with factory() as s:
                async with _get_activity_lock(team_id, athlete_id):
                    r = await s.execute(
                        select(Activity).where(
                            Activity.athlete_id == athlete_id,
                            Activity.start_time >= base_time - _DUPLICATE_WINDOW,
                            Activity.start_time <= base_time + _DUPLICATE_WINDOW,
                        )
                    )
                    if r.scalar_one_or_none() is None:
                        s.add(Activity(
                            athlete_id=athlete_id,
                            start_time=base_time,
                            duration_s=5009,
                            status="pending",
                        ))
                        await s.flush()
                    await s.commit()

        await asyncio.gather(first_task(), second_task())

        async with factory() as s:
            r = await s.execute(select(Activity).where(Activity.athlete_id == athlete_id))
            activities = r.scalars().all()

        assert len(activities) == 1, (
            f"Race condition created {len(activities)} duplicate Activity records "
            f"for the same workout (expected 1). "
            f"The asyncio lock must guard both the flush and the commit."
        )


# ── The in-process lock cache ──────────────────────────────────────────────────


class TestActivityLockCache:
    """The lock dict used to grow forever and was keyed by athlete (issue #50)."""

    def _clear(self):
        from backend.app.services.provider_sync import _activity_creation_locks

        _activity_creation_locks.clear()

    async def test_the_same_athlete_gets_the_same_lock(self):
        from backend.app.services.provider_sync import _get_activity_lock

        self._clear()
        assert _get_activity_lock("u", "a") is _get_activity_lock("u", "a")
        assert _get_activity_lock("u", "a") is not _get_activity_lock("u", "b")

    async def test_the_cache_is_bounded(self):
        from backend.app.services.provider_sync import (
            _MAX_ACTIVITY_LOCKS,
            _activity_creation_locks,
            _get_activity_lock,
        )

        self._clear()
        for i in range(_MAX_ACTIVITY_LOCKS * 2):
            _get_activity_lock("u", f"athlete-{i}")
        assert len(_activity_creation_locks) <= _MAX_ACTIVITY_LOCKS

    async def test_a_held_lock_is_never_evicted(self):
        """Evicting a held lock would hand one athlete two locks at once."""
        from backend.app.services.provider_sync import (
            _MAX_ACTIVITY_LOCKS,
            _activity_creation_locks,
            _get_activity_lock,
        )

        self._clear()
        held = _get_activity_lock("u", "busy")
        async with held:
            for i in range(_MAX_ACTIVITY_LOCKS * 2):
                _get_activity_lock("u", f"athlete-{i}")
            assert _activity_creation_locks[("u", "busy")][1] is held
            assert _get_activity_lock("u", "busy") is held
        self._clear()


# ── Throttling, repair and the backfill's limits (issue #67) ──────────────────


def _status_error(status: int, headers: dict | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://provider.test/activities/1/streams")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


def _client(**overrides) -> MagicMock:
    """A provider client whose FIT download is absent and whose streams arrive."""
    client = MagicMock()
    client.list_activities = AsyncMock(side_effect=[[_norm()], []])
    client.download_fit_file = AsyncMock(return_value=None)
    client.get_activity_streams = AsyncMock(
        return_value={"power": [200, 210, 220], "heartrate": [140, 145, 150]}
    )
    for key, value in overrides.items():
        setattr(client, key, value)
    return client


async def _sync(athlete, session, client, provider: str = "strava"):
    conn = _make_connection(athlete, provider=provider)
    with patch(
        "backend.app.services.provider_sync.PROVIDERS",
        {provider: MagicMock(return_value=client)},
    ):
        return await sync_provider_activities(
            athlete, conn, session, user_id=_TEAM_ID, access_token=_ACCESS_TOKEN
        )


async def _stream_types(session, activity_id: str) -> set[str]:
    from backend.app.models.user_orm import ActivityStream

    result = await session.execute(
        select(ActivityStream).where(ActivityStream.activity_id == activity_id)
    )
    return {s.stream_type for s in result.scalars()}


class TestThrottling:
    async def test_a_mid_import_429_persists_no_hollow_activity(self, session):
        """The activity behind the throttle is not imported at all.

        Importing it anyway is the data-loss bug: it lands with no power, HR or
        cadence, and the skip at the top of the loop then steps over it forever.
        """
        athlete = await _make_athlete(session, user_id="throttle-1")
        first = _norm("act-1", start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc))
        second = _norm("act-2", start_time=datetime(2024, 6, 2, 10, 0, tzinfo=timezone.utc))

        client = _client(
            list_activities=AsyncMock(side_effect=[[first, second], []]),
            get_activity_streams=AsyncMock(
                side_effect=[{"power": [200, 210, 220]}, _status_error(429)]
            ),
        )
        count, earliest = await _sync(athlete, session, client)

        assert count == 1
        assert earliest == date(2024, 6, 1)

        acts = (
            await session.execute(select(Activity).where(Activity.athlete_id == athlete.id))
        ).scalars().all()
        assert len(acts) == 1, "the throttled activity must not be persisted"
        assert await _stream_types(session, acts[0].id) >= {"power"}

        srcs = (await session.execute(select(ActivitySource))).scalars().all()
        assert [s.external_id for s in srcs] == ["act-1"]

    async def test_a_resync_imports_what_the_throttle_stopped(self, session):
        """The stopped import resumes rather than needing anything of the athlete."""
        athlete = await _make_athlete(session, user_id="throttle-2")
        first = _norm("act-1", start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc))
        second = _norm("act-2", start_time=datetime(2024, 6, 2, 10, 0, tzinfo=timezone.utc))

        await _sync(
            athlete,
            session,
            _client(
                list_activities=AsyncMock(side_effect=[[first, second], []]),
                get_activity_streams=AsyncMock(
                    side_effect=[{"power": [200, 210, 220]}, _status_error(429)]
                ),
            ),
        )

        count, _ = await _sync(
            athlete,
            session,
            _client(list_activities=AsyncMock(side_effect=[[first, second], []])),
        )

        assert count == 1  # act-1 was already here; act-2 is the new one
        acts = (
            await session.execute(
                select(Activity).where(Activity.athlete_id == athlete.id)
            )
        ).scalars().all()
        assert len(acts) == 2
        for act in acts:
            assert await _stream_types(session, act.id) >= {"power"}

    async def test_a_throttled_prefetch_stops_the_sync(self, session):
        """The FIT prefetch is a fetch like any other — a 429 there is not "no FIT"."""
        athlete = await _make_athlete(session, user_id="throttle-3")
        client = _client(download_fit_file=AsyncMock(side_effect=_status_error(429)))

        count, _ = await _sync(athlete, session, client, provider="wahoo")

        assert count == 0
        acts = (
            await session.execute(select(Activity).where(Activity.athlete_id == athlete.id))
        ).scalars().all()
        assert acts == []
        client.get_activity_streams.assert_not_awaited()

    async def test_a_throttled_attach_leaves_no_settled_source_behind(self, session):
        """The attach path decides priority from the FIT it prefetches.

        Deciding it on an answer the provider never gave would attach the source,
        record it as needing nothing, and leave a device file that outranks
        what is on the activity permanently unfetched.
        """
        athlete = await _make_athlete(session, user_id="throttle-7")
        base_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
        act = Activity(
            athlete_id=athlete.id,
            start_time=base_time,
            duration_s=3600,
            status="processed",
        )
        session.add(act)
        await session.flush()
        session.add(
            ActivitySource(
                activity_id=act.id,
                provider="strava",
                external_id="s-1",
                streams_fetched_at=base_time,
            )
        )
        await session.commit()

        client = _client(
            list_activities=AsyncMock(
                side_effect=[[_norm("w-1", "wahoo", base_time)], []]
            ),
            download_fit_file=AsyncMock(side_effect=_status_error(429)),
        )
        count, _ = await _sync(athlete, session, client, provider="wahoo")

        assert count == 0
        providers = {
            src.provider
            for src in (await session.execute(select(ActivitySource))).scalars()
        }
        assert providers == {"strava"}

    async def test_retry_after_is_honoured(self, session):
        """A provider that says when to come back is taken at its word, once."""
        athlete = await _make_athlete(session, user_id="throttle-4")
        client = _client(
            get_activity_streams=AsyncMock(
                side_effect=[
                    _status_error(429, {"Retry-After": "3"}),
                    {"power": [200, 210, 220]},
                ]
            )
        )

        with patch(
            "backend.app.services.providers.throttling.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep:
            count, _ = await _sync(athlete, session, client)

        sleep.assert_awaited_once_with(3.0)
        assert count == 1
        act = (
            await session.execute(select(Activity).where(Activity.athlete_id == athlete.id))
        ).scalar_one()
        assert await _stream_types(session, act.id) >= {"power"}

    async def test_a_failed_stream_fetch_leaves_the_source_unfinished(self, session):
        """A failure that is not the provider's final word must not settle anything."""
        athlete = await _make_athlete(session, user_id="throttle-5")
        client = _client(
            get_activity_streams=AsyncMock(side_effect=httpx.ReadTimeout("slow"))
        )

        await _sync(athlete, session, client)

        src = (await session.execute(select(ActivitySource))).scalar_one()
        assert src.streams_fetched_at is None

    async def test_a_404_on_streams_settles_the_source(self, session):
        """Strava answers a manual entry's streams with a 404 — every time.

        Retrying that on every sync forever is the amplification this is meant to
        avoid, so a final answer settles the source even though no data came back.
        """
        athlete = await _make_athlete(session, user_id="throttle-6")
        client = _client(get_activity_streams=AsyncMock(side_effect=_status_error(404)))

        await _sync(athlete, session, client)

        src = (await session.execute(select(ActivitySource))).scalar_one()
        assert src.streams_fetched_at is not None


class TestRepairingIncompleteImports:
    async def test_a_source_with_no_streams_is_repaired_not_skipped(self, session):
        """The acceptance criterion of issue #67: hollow imports must heal.

        This is the shape of every activity imported behind a throttle before the
        fix, and of every row the migration could not vouch for.
        """
        athlete = await _make_athlete(session, user_id="repair-1")
        act = Activity(
            athlete_id=athlete.id,
            start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
            duration_s=3600,
            status="processed",
        )
        session.add(act)
        await session.flush()
        session.add(
            ActivitySource(activity_id=act.id, provider="strava", external_id="act-1")
        )
        await session.commit()

        client = _client()
        await _sync(athlete, session, client)

        client.get_activity_streams.assert_awaited_once()
        assert await _stream_types(session, act.id) >= {"power", "heartrate"}
        src = (await session.execute(select(ActivitySource))).scalar_one()
        assert src.streams_fetched_at is not None

    async def test_a_complete_source_is_never_re_fetched(self, session):
        athlete = await _make_athlete(session, user_id="repair-2")
        act = Activity(
            athlete_id=athlete.id,
            start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
            duration_s=3600,
            status="processed",
        )
        session.add(act)
        await session.flush()
        session.add(
            ActivitySource(
                activity_id=act.id,
                provider="strava",
                external_id="act-1",
                streams_fetched_at=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
            )
        )
        await session.commit()

        client = _client()
        count, _ = await _sync(athlete, session, client)

        assert count == 0
        client.get_activity_streams.assert_not_awaited()
        client.download_fit_file.assert_not_awaited()

    async def test_repair_does_not_overwrite_a_better_source(self, session):
        """A source that lost the priority contest has nothing to repair.

        Refilling from it would replace a device file's streams with a summary
        API's, which is the wrong direction — so it is settled, not fetched.
        """
        athlete = await _make_athlete(session, user_id="repair-3")
        act = Activity(
            athlete_id=athlete.id,
            start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
            duration_s=3600,
            status="processed",
        )
        session.add(act)
        await session.flush()
        session.add(
            ActivitySource(
                activity_id=act.id,
                provider="upload",
                external_id=None,
                fit_file_path="/tmp/does-not-need-to-exist.fit",
                streams_fetched_at=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
            )
        )
        session.add(
            ActivitySource(activity_id=act.id, provider="strava", external_id="act-1")
        )
        await session.commit()

        client = _client()
        await _sync(athlete, session, client)

        client.get_activity_streams.assert_not_awaited()
        strava_src = (
            await session.execute(
                select(ActivitySource).where(ActivitySource.provider == "strava")
            )
        ).scalar_one()
        assert strava_src.streams_fetched_at is not None

    async def test_a_source_attached_below_the_winner_needs_no_fetch(self, session):
        """The attach path settles what it decides not to populate.

        Otherwise every later sync would come back to re-fetch data the priority
        rules have already ruled out ever reading.
        """
        athlete = await _make_athlete(session, user_id="repair-4")
        base_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)

        upload = Activity(
            athlete_id=athlete.id,
            start_time=base_time,
            duration_s=3600,
            status="processed",
        )
        session.add(upload)
        await session.flush()
        session.add(
            ActivitySource(
                activity_id=upload.id,
                provider="upload",
                external_id=None,
                fit_file_path="/tmp/does-not-need-to-exist.fit",
                streams_fetched_at=base_time,
            )
        )
        await session.commit()

        client = _client(
            list_activities=AsyncMock(
                side_effect=[[_norm("act-1", start_time=base_time)], []]
            )
        )
        await _sync(athlete, session, client)

        strava_src = (
            await session.execute(
                select(ActivitySource).where(ActivitySource.provider == "strava")
            )
        ).scalar_one()
        assert strava_src.streams_fetched_at is not None
        client.get_activity_streams.assert_not_awaited()


class TestBackfillLimits:
    async def test_the_page_loop_is_bounded(self, session):
        """A provider that never returns an empty page must not spin forever."""
        athlete = await _make_athlete(session, user_id="bounded-1")
        act = Activity(
            athlete_id=athlete.id,
            start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
            duration_s=3600,
            status="processed",
        )
        session.add(act)
        await session.flush()
        session.add(
            ActivitySource(
                activity_id=act.id,
                provider="strava",
                external_id="act-1",
                streams_fetched_at=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
            )
        )
        await session.commit()

        # The same already-imported activity on every page: a provider stuck on
        # page 1 is exactly the bug the cap exists for, and it keeps the test to
        # one cheap skip per page. The cap is patched down so the test exercises
        # the mechanism rather than the number.
        client = _client(list_activities=AsyncMock(return_value=[_norm("act-1")]))
        with patch("backend.app.services.provider_sync._MAX_SYNC_PAGES", 5):
            count, _ = await _sync(athlete, session, client)

        assert count == 0
        assert client.list_activities.await_count == 5

    async def test_the_walk_is_bounded_in_activities_not_pages(self, session):
        """A page is not a fixed amount of anything.

        Strava lists 200 at a time and Wahoo 30, so a page cap alone would mean
        20 000 workouts on one provider and 3 000 on the other — and 3 000 is
        inside a real riding history, which a bound meant as an impossibility
        must never be.
        """
        athlete = await _make_athlete(session, user_id="bounded-2")
        act = Activity(
            athlete_id=athlete.id,
            start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
            duration_s=3600,
            status="processed",
        )
        session.add(act)
        await session.flush()
        session.add(
            ActivitySource(
                activity_id=act.id,
                provider="strava",
                external_id="act-1",
                streams_fetched_at=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
            )
        )
        await session.commit()

        # Three activities a page, a cap of seven: the walk must stop on the page
        # that crosses it rather than on a page count.
        client = _client(
            list_activities=AsyncMock(
                return_value=[_norm("act-1"), _norm("act-1"), _norm("act-1")]
            )
        )
        with patch("backend.app.services.provider_sync._MAX_SYNC_ACTIVITIES", 7):
            await _sync(athlete, session, client)

        assert client.list_activities.await_count == 3  # 3, 6, 9 → stops at 9 ≥ 7

    async def test_a_second_concurrent_sync_does_not_start(self, session):
        """One backfill per (user, provider): the quota is the instance's, not the user's."""
        from datetime import timedelta as _timedelta

        from backend.app.db import leases
        from backend.app.models.user_orm import SyncLease

        athlete = await _make_athlete(session, user_id="single-flight-1")
        held = await leases.acquire(
            session,
            SyncLease,
            "provider-sync:strava",
            ttl=_timedelta(minutes=15),
            wait=0.0,
        )
        assert held is not None

        client = _client()
        try:
            count, earliest = await _sync(athlete, session, client)
        finally:
            await leases.release(session, SyncLease, "provider-sync:strava", held)

        assert (count, earliest) == (0, None)
        client.list_activities.assert_not_awaited()

    async def test_the_lease_is_released_for_the_next_sync(self, session):
        athlete = await _make_athlete(session, user_id="single-flight-2")

        await _sync(athlete, session, _client())

        second = _client(list_activities=AsyncMock(side_effect=[[_norm("act-2")], []]))
        count, _ = await _sync(athlete, session, second)

        second.list_activities.assert_awaited()
        assert count == 1

    async def test_the_lease_is_released_when_the_sync_raises(self, session):
        athlete = await _make_athlete(session, user_id="single-flight-3")
        boom = _client(list_activities=AsyncMock(side_effect=RuntimeError("boom")))

        with pytest.raises(RuntimeError):
            await _sync(athlete, session, boom)

        recovered = _client()
        count, _ = await _sync(athlete, session, recovered)
        assert count == 1


class TestRepairCountsAsAnUpdate:
    async def test_a_repair_reports_its_date_for_recalculation(self, session):
        """A repair restates load and intensity, so the metrics behind them move.

        The caller only recalculates when the sync reports a count and a date;
        a repair that reported neither would leave the athlete's fitness and
        fatigue sitting on the hollow ride's figures.
        """
        athlete = await _make_athlete(session, user_id="repair-5")
        athlete.ftp = 250
        act = Activity(
            athlete_id=athlete.id,
            start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
            duration_s=3600,
            status="processed",
        )
        session.add(act)
        await session.flush()
        session.add(
            ActivitySource(activity_id=act.id, provider="strava", external_id="act-1")
        )
        await session.commit()

        count, earliest = await _sync(
            athlete,
            session,
            _client(
                get_activity_streams=AsyncMock(return_value={"power": [200] * 3600})
            ),
        )

        assert count == 1
        assert earliest == date(2024, 6, 1)
        await session.refresh(act)
        assert act.load is not None

    async def test_a_settled_source_is_not_counted_as_an_update(self, session):
        """Deciding a source needs no data is not a change to the athlete's history."""
        athlete = await _make_athlete(session, user_id="repair-6")
        act = Activity(
            athlete_id=athlete.id,
            start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
            duration_s=3600,
            status="processed",
        )
        session.add(act)
        await session.flush()
        session.add(
            ActivitySource(
                activity_id=act.id,
                provider="upload",
                external_id=None,
                fit_file_path="/tmp/does-not-need-to-exist.fit",
                streams_fetched_at=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
            )
        )
        session.add(
            ActivitySource(activity_id=act.id, provider="strava", external_id="act-1")
        )
        await session.commit()

        count, earliest = await _sync(athlete, session, _client())

        assert (count, earliest) == (0, None)


class TestLosingTheSyncLease:
    async def test_the_walk_stops_the_moment_the_lease_is_gone(self, session):
        """A lost lease means someone else is importing this history now.

        Carrying on would be exactly the duplicate backfill the lease exists to
        prevent. Checked per activity rather than per page: a Strava page is 200
        activities, so a page-boundary check would let a whole page of duplicated
        work through after the deadline had already lapsed.
        """
        from backend.app.db import leases

        athlete = await _make_athlete(session, user_id="lease-lost-1")
        activities = [
            _norm("act-1", start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)),
            _norm("act-2", start_time=datetime(2024, 6, 2, 10, 0, tzinfo=timezone.utc)),
            _norm("act-3", start_time=datetime(2024, 6, 3, 10, 0, tzinfo=timezone.utc)),
        ]
        client = _client(list_activities=AsyncMock(side_effect=[activities, []]))

        real_renew = leases.renew
        calls = 0

        async def _lost_after_one(*args, **kwargs):
            # The real write still happens; only the answer is a lie, so the test
            # fails if the `break` goes away rather than if the mock does.
            nonlocal calls
            await real_renew(*args, **kwargs)
            calls += 1
            return calls == 1

        with patch.object(leases, "renew", _lost_after_one):
            count, _ = await _sync(athlete, session, client)

        # Stopped inside the page, not at the end of it.
        assert count == 1
        acts = (
            await session.execute(select(Activity).where(Activity.athlete_id == athlete.id))
        ).scalars().all()
        assert len(acts) == 1

    async def test_a_lease_lost_before_the_first_activity_imports_nothing(self, session):
        from backend.app.db import leases

        athlete = await _make_athlete(session, user_id="lease-lost-2")
        client = _client()

        real_renew = leases.renew

        async def _lost(*args, **kwargs):
            await real_renew(*args, **kwargs)
            return False

        with patch.object(leases, "renew", _lost):
            count, earliest = await _sync(athlete, session, client)

        assert (count, earliest) == (0, None)
        client.get_activity_streams.assert_not_awaited()


class TestDurationCorrection:
    """The provider's `moving_time` beating what is already stored.

    Both branches recompute the ride's load, which is what makes this more than
    a cosmetic field update: an activity whose duration shrinks by half has to
    stop claiming the training stress of the longer one.
    """

    async def _seed(self, session, athlete, **activity_fields) -> Activity:
        act = Activity(
            athlete_id=athlete.id,
            start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
            duration_s=7200,
            status="processed",
            **activity_fields,
        )
        session.add(act)
        await session.flush()
        session.add(
            ActivitySource(
                activity_id=act.id,
                provider="strava",
                external_id="act-1",
                streams_fetched_at=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
            )
        )
        await session.commit()
        return act

    async def test_a_shorter_duration_restates_load_from_power(self, session):
        athlete = await _make_athlete(session, user_id="duration-1")
        athlete.ftp = 250
        athlete.max_hr = 190
        act = await self._seed(session, athlete, weighted_power=200.0, avg_hr=150.0)
        before = act.load

        client = _client()  # `_norm` reports 3600 s, half of what is stored
        count, _ = await _sync(athlete, session, client)

        assert count == 0  # a correction is not an import
        await session.refresh(act)
        assert act.duration_s == 3600
        assert act.load != before
        assert act.intensity is not None

    async def test_a_shorter_duration_restates_load_from_hr_alone(self, session):
        """No power on the ride, so the HR branch has to carry the recompute."""
        athlete = await _make_athlete(session, user_id="duration-2")
        athlete.max_hr = 190
        act = await self._seed(session, athlete, avg_hr=150.0)

        await _sync(athlete, session, _client())

        await session.refresh(act)
        assert act.duration_s == 3600
        assert act.load is not None

    async def test_a_longer_duration_is_left_alone(self, session):
        """Only `moving_time` beating `elapsed_time` is a correction."""
        athlete = await _make_athlete(session, user_id="duration-3")
        act = Activity(
            athlete_id=athlete.id,
            start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
            duration_s=1800,
            status="processed",
        )
        session.add(act)
        await session.flush()
        session.add(
            ActivitySource(
                activity_id=act.id,
                provider="strava",
                external_id="act-1",
                streams_fetched_at=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
            )
        )
        await session.commit()

        await _sync(athlete, session, _client())

        await session.refresh(act)
        assert act.duration_s == 1800


class TestAServerErrorIsNotAThrottle:
    """Review of #143: a 5xx is about one request, a 429 about the connection.

    Treating them alike made a single unservable ride a permanent wall — the walk
    is newest-first and restarts from the newest each sync, so everything older
    than the poison activity became unreachable forever.
    """

    async def test_one_permanently_500_activity_does_not_block_older_ones(self, session):
        athlete = await _make_athlete(session, user_id="poison-1")
        listing = [
            _norm("act-1", start_time=datetime(2024, 6, 3, 10, 0, tzinfo=timezone.utc)),
            _norm("act-2", start_time=datetime(2024, 6, 2, 10, 0, tzinfo=timezone.utc)),
            _norm("act-3", start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)),
        ]

        async def streams(_token, ext_id):
            if ext_id == "act-2":
                raise _status_error(500)
            return {"power": [200, 210, 220]}

        await _sync(
            athlete,
            session,
            _client(
                list_activities=AsyncMock(side_effect=[list(listing), []]),
                get_activity_streams=AsyncMock(side_effect=streams),
            ),
        )

        srcs = (await session.execute(select(ActivitySource))).scalars().all()
        assert sorted(s.external_id for s in srcs) == ["act-1", "act-2", "act-3"]

        by_id = {s.external_id: s for s in srcs}
        assert by_id["act-1"].streams_fetched_at is not None
        assert by_id["act-3"].streams_fetched_at is not None
        # …and the one that could not be served is left for repair, not settled.
        assert by_id["act-2"].streams_fetched_at is None

    async def test_a_429_still_stops_the_walk(self, session):
        """The distinction has to cut one way only."""
        athlete = await _make_athlete(session, user_id="poison-2")
        listing = [
            _norm("act-1", start_time=datetime(2024, 6, 3, 10, 0, tzinfo=timezone.utc)),
            _norm("act-2", start_time=datetime(2024, 6, 2, 10, 0, tzinfo=timezone.utc)),
            _norm("act-3", start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)),
        ]

        async def streams(_token, ext_id):
            if ext_id == "act-2":
                raise _status_error(429)
            return {"power": [200, 210, 220]}

        await _sync(
            athlete,
            session,
            _client(
                list_activities=AsyncMock(side_effect=[list(listing), []]),
                get_activity_streams=AsyncMock(side_effect=streams),
            ),
        )

        srcs = (await session.execute(select(ActivitySource))).scalars().all()
        assert [s.external_id for s in srcs] == ["act-1"]

    async def test_a_provider_failing_everything_stops_the_walk(self, session):
        """One bad ride is not a reason to stop; a provider that is down is.

        Every failure still costs a round trip, so a walk that carried on through
        thousands of them would be the amplification the issue is about.
        """
        athlete = await _make_athlete(session, user_id="poison-3")
        listing = [
            _norm(
                f"act-{i}",
                start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
                + timedelta(days=i),
            )
            for i in range(10)
        ]
        client = _client(
            list_activities=AsyncMock(side_effect=[listing, []]),
            get_activity_streams=AsyncMock(side_effect=_status_error(503)),
        )

        with patch(
            "backend.app.services.provider_sync._MAX_CONSECUTIVE_UNRESOLVED", 3
        ):
            await _sync(athlete, session, client)

        # Stopped after the run of failures rather than walking all ten — and
        # without spending one more request to find that out, because the run is
        # checked before each activity rather than after it.
        assert client.get_activity_streams.await_count == 3
        srcs = (await session.execute(select(ActivitySource))).scalars().all()
        assert all(s.streams_fetched_at is None for s in srcs)

    async def test_a_success_resets_the_run(self, session):
        """Scattered bad rides must never accumulate into a stop."""
        athlete = await _make_athlete(session, user_id="poison-4")
        listing = [
            _norm(
                f"act-{i}",
                start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
                + timedelta(days=i),
            )
            for i in range(9)
        ]

        async def streams(_token, ext_id):
            # Every third one fails: never three in a row.
            if int(ext_id.split("-")[1]) % 3:
                raise _status_error(500)
            return {"power": [200, 210, 220]}

        client = _client(
            list_activities=AsyncMock(side_effect=[listing, []]),
            get_activity_streams=AsyncMock(side_effect=streams),
        )
        with patch(
            "backend.app.services.provider_sync._MAX_CONSECUTIVE_UNRESOLVED", 3
        ):
            await _sync(athlete, session, client)

        srcs = (await session.execute(select(ActivitySource))).scalars().all()
        assert len(srcs) == 9, "the walk must reach every activity"


class TestAGuessIsNotAnAnswer:
    """Review of #143: a FIT we failed to reach is not a FIT that is not there."""

    async def _seed_populated_strava(self, session, athlete, base):
        act = Activity(
            athlete_id=athlete.id, start_time=base, duration_s=3600, status="processed"
        )
        session.add(act)
        await session.flush()
        session.add(
            ActivitySource(
                activity_id=act.id,
                provider="strava",
                external_id="s-1",
                streams_fetched_at=base,
            )
        )
        await session.commit()
        return act

    async def test_a_timeout_on_the_attach_prefetch_leaves_the_source_open(self, session):
        """Otherwise the device file that would have won is buried permanently.

        Wahoo-with-FIT outranks Strava; Wahoo-without loses to it. Deciding that
        contest on a `ConnectTimeout` and then stamping the source means the
        higher-priority file is never fetched again.
        """
        athlete = await _make_athlete(session, user_id="guess-1")
        base = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
        await self._seed_populated_strava(session, athlete, base)

        wahoo = _client(
            list_activities=AsyncMock(side_effect=[[_norm("w-1", "wahoo", base)], []]),
            download_fit_file=AsyncMock(side_effect=httpx.ConnectTimeout("boom")),
        )
        await _sync(athlete, session, wahoo, provider="wahoo")

        src = (
            await session.execute(
                select(ActivitySource).where(ActivitySource.provider == "wahoo")
            )
        ).scalar_one()
        assert src.streams_fetched_at is None

    async def test_a_definitive_absence_still_settles_the_source(self, session):
        """The other half: a provider that says "no file" has answered."""
        athlete = await _make_athlete(session, user_id="guess-2")
        base = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
        await self._seed_populated_strava(session, athlete, base)

        wahoo = _client(
            list_activities=AsyncMock(side_effect=[[_norm("w-1", "wahoo", base)], []]),
            download_fit_file=AsyncMock(return_value=None),
        )
        await _sync(athlete, session, wahoo, provider="wahoo")

        src = (
            await session.execute(
                select(ActivitySource).where(ActivitySource.provider == "wahoo")
            )
        ).scalar_one()
        assert src.streams_fetched_at is not None

    async def test_a_repair_is_not_spent_on_a_failed_lookup(self, session):
        """A source gets one repair attempt — not one wasted on a timeout."""
        athlete = await _make_athlete(session, user_id="guess-3")
        base = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
        act = Activity(
            athlete_id=athlete.id, start_time=base, duration_s=3600, status="processed"
        )
        session.add(act)
        await session.flush()
        session.add(
            ActivitySource(
                activity_id=act.id,
                provider="upload",
                external_id=None,
                fit_file_path="/tmp/does-not-need-to-exist.fit",
                streams_fetched_at=base,
            )
        )
        session.add(
            ActivitySource(activity_id=act.id, provider="wahoo", external_id="w-1")
        )
        await session.commit()

        wahoo = _client(
            list_activities=AsyncMock(side_effect=[[_norm("w-1", "wahoo", base)], []]),
            download_fit_file=AsyncMock(side_effect=httpx.ReadTimeout("slow")),
        )
        await _sync(athlete, session, wahoo, provider="wahoo")

        src = (
            await session.execute(
                select(ActivitySource).where(ActivitySource.provider == "wahoo")
            )
        ).scalar_one()
        assert src.streams_fetched_at is None


# ── Recording what a run did, and resuming it (issue #68) ─────────────────────


async def _state(session, provider: str = "strava"):
    from backend.app.services.sync_state import get_state

    return await get_state(session, provider)


async def _imported_source(session, athlete, ext_id: str, start_time: datetime):
    """An activity already fully imported from Strava, as a finished run leaves it."""
    act = Activity(
        athlete_id=athlete.id,
        start_time=start_time,
        duration_s=3600,
        status="processed",
    )
    session.add(act)
    await session.flush()
    session.add(
        ActivitySource(
            activity_id=act.id,
            provider="strava",
            external_id=ext_id,
            streams_fetched_at=start_time,
        )
    )
    await session.commit()
    return act


class TestRecordingHowARunEnded:
    """The four ways to stop are four different recorded facts, not one log line."""

    async def test_a_finished_walk_records_a_completion(self, session):
        athlete = await _make_athlete(session, user_id="state-complete")
        await _sync(
            athlete,
            session,
            _client(
                list_activities=AsyncMock(
                    side_effect=[[_norm("act-1", start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc))], []]
                )
            ),
        )

        state = await _state(session)
        assert state.status == "completed"
        assert state.stop_reason is None
        assert state.imported == 1
        assert state.listed == 1
        assert state.oldest_seen_on == date(2024, 6, 1)
        # Nothing to resume: the next run starts from the front and walks the lot.
        assert state.resume_page is None

    async def test_a_throttle_stop_is_recorded_with_its_reason(self, session):
        athlete = await _make_athlete(session, user_id="state-throttle")
        first = _norm("act-1", start_time=datetime(2024, 6, 2, 10, 0, tzinfo=timezone.utc))
        second = _norm("act-2", start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc))

        await _sync(
            athlete,
            session,
            _client(
                list_activities=AsyncMock(side_effect=[[first, second], []]),
                get_activity_streams=AsyncMock(
                    side_effect=[{"power": [200, 210, 220]}, _status_error(429)]
                ),
            ),
        )

        state = await _state(session)
        assert state.status == "stopped"
        assert state.stop_reason == "throttled"
        assert "429" in state.stop_detail
        assert state.imported == 1
        # The page it gave up on, so the next run can continue from there.
        assert state.resume_page == 1
        assert state.repeat_count == 1
        assert state.repeat_since is not None

    async def test_the_safety_limit_stop_is_recorded(self, session):
        athlete = await _make_athlete(session, user_id="state-bound")
        await _imported_source(
            session, athlete, "act-1", datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
        )

        client = _client(list_activities=AsyncMock(return_value=[_norm("act-1")]))
        with patch("backend.app.services.provider_sync._MAX_SYNC_PAGES", 3):
            await _sync(athlete, session, client)

        state = await _state(session)
        assert state.status == "stopped"
        assert state.stop_reason == "safety_limit"
        assert state.resume_page == 4  # the page the cap refused to list

    async def test_a_provider_serving_nothing_is_recorded_as_an_outage(self, session):
        athlete = await _make_athlete(session, user_id="state-outage")
        activities = [
            _norm(f"act-{i}", start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
                  + timedelta(days=i))
            for i in range(4)
        ]
        client = _client(
            list_activities=AsyncMock(side_effect=[activities, []]),
            get_activity_streams=AsyncMock(side_effect=_status_error(500)),
        )
        with patch("backend.app.services.provider_sync._MAX_CONSECUTIVE_UNRESOLVED", 2):
            await _sync(athlete, session, client)

        state = await _state(session)
        assert state.status == "stopped"
        assert state.stop_reason == "provider_outage"
        assert "detail data" in state.stop_detail

    async def test_a_lost_lease_is_recorded(self, session):
        from backend.app.db import leases as lease_mod

        athlete = await _make_athlete(session, user_id="state-lease")
        client = _client(
            list_activities=AsyncMock(side_effect=[[_norm("act-1"), _norm("act-2")], []])
        )

        async def _lost(*args, **kwargs):
            return False

        with patch.object(lease_mod, "renew", _lost):
            await _sync(athlete, session, client)

        state = await _state(session)
        assert state.status == "stopped"
        assert state.stop_reason == "lease_lost"
        assert state.imported == 0

    async def test_an_exception_is_recorded_as_an_error(self, session):
        athlete = await _make_athlete(session, user_id="state-error")
        boom = _client(list_activities=AsyncMock(side_effect=RuntimeError("boom")))

        with pytest.raises(RuntimeError):
            await _sync(athlete, session, boom)

        state = await _state(session)
        assert state.status == "stopped"
        assert state.stop_reason == "error"
        assert "boom" in state.stop_detail

    async def test_a_refused_duplicate_run_records_nothing(self, session):
        """"Already running" is not this run's outcome to record."""
        from datetime import timedelta as _timedelta

        from backend.app.db import leases
        from backend.app.models.user_orm import SyncLease

        athlete = await _make_athlete(session, user_id="state-duplicate")
        held = await leases.acquire(
            session, SyncLease, "provider-sync:strava",
            ttl=_timedelta(minutes=15), wait=0.0,
        )
        try:
            await _sync(athlete, session, _client())
        finally:
            await leases.release(session, SyncLease, "provider-sync:strava", held)

        assert await _state(session) is None


class TestSeeingARepeatAsARepeat:
    """A sync stopping in the same place three runs running is one fact, not three."""

    async def test_the_same_stop_twice_counts_as_a_repeat(self, session):
        athlete = await _make_athlete(session, user_id="repeat-1")
        norm = _norm("act-1", start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc))

        for _ in range(3):
            await _sync(
                athlete,
                session,
                _client(
                    list_activities=AsyncMock(side_effect=[[norm], []]),
                    get_activity_streams=AsyncMock(side_effect=_status_error(429)),
                ),
            )

        state = await _state(session)
        assert state.stop_reason == "throttled"
        assert state.repeat_count == 3

    async def test_a_different_stop_starts_a_new_streak(self, session):
        athlete = await _make_athlete(session, user_id="repeat-2")
        norm = _norm("act-1", start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc))

        await _sync(
            athlete,
            session,
            _client(
                list_activities=AsyncMock(side_effect=[[norm], []]),
                get_activity_streams=AsyncMock(side_effect=_status_error(429)),
            ),
        )
        with pytest.raises(RuntimeError):
            await _sync(
                athlete,
                session,
                _client(list_activities=AsyncMock(side_effect=RuntimeError("boom"))),
            )

        state = await _state(session)
        assert state.stop_reason == "error"
        assert state.repeat_count == 1

    async def test_a_completion_clears_the_streak(self, session):
        athlete = await _make_athlete(session, user_id="repeat-3")
        norm = _norm("act-1", start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc))

        await _sync(
            athlete,
            session,
            _client(
                list_activities=AsyncMock(side_effect=[[norm], []]),
                get_activity_streams=AsyncMock(side_effect=_status_error(429)),
            ),
        )
        await _sync(
            athlete,
            session,
            _client(list_activities=AsyncMock(side_effect=[[norm], []])),
        )

        state = await _state(session)
        assert state.status == "completed"
        assert state.stop_reason is None
        assert state.repeat_count == 0
        assert state.resume_page is None


class TestResumingWhereItStopped:
    """A stopped run leaves the next one somewhere to start."""

    async def test_the_next_run_skips_the_pages_it_already_walked(self, session):
        """The whole point: fresh quota goes on new history, not on re-listing old.

        Three pages of activities that are already imported, then a fourth the
        run never reaches because it is throttled. The run after it walks page 1,
        finds nothing to do, and jumps straight to where the stop was recorded
        rather than paying for pages 2 and 3 again.
        """
        athlete = await _make_athlete(session, user_id="resume-1")
        base = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
        pages = {
            1: [_norm("p1", start_time=base)],
            2: [_norm("p2", start_time=base - timedelta(days=1))],
            3: [_norm("p3", start_time=base - timedelta(days=2))],
            4: [_norm("p4", start_time=base - timedelta(days=3))],
        }
        for page, (norm,) in pages.items():
            if page < 4:
                await _imported_source(session, athlete, norm.external_id, norm.start_time)

        async def _list(_token, page):
            return pages.get(page, [])

        # Run one: reaches the new activity on page 4 and is thrown out.
        await _sync(
            athlete,
            session,
            _client(
                list_activities=AsyncMock(side_effect=_list),
                get_activity_streams=AsyncMock(side_effect=_status_error(429)),
            ),
        )
        assert (await _state(session)).resume_page == 4

        # Run two: page 1 settles the front, then the jump.
        second = _client(list_activities=AsyncMock(side_effect=_list))
        count, _ = await _sync(athlete, session, second)

        listed_pages = [call.args[1] for call in second.list_activities.await_args_list]
        assert listed_pages == [1, 3, 4, 5], "pages 2 was already walked; 3 is the overlap"
        assert count == 1  # the activity the throttle stopped

    async def test_a_new_ride_at_the_front_is_never_skipped(self, session):
        """The front sweep runs before the jump, so new history still lands."""
        athlete = await _make_athlete(session, user_id="resume-2")
        base = datetime(2024, 6, 10, 10, 0, tzinfo=timezone.utc)
        old = _norm("old", start_time=base - timedelta(days=30))
        await _imported_source(session, athlete, "old", old.start_time)
        fresh = _norm("fresh", start_time=base)

        async def _list(_token, page):
            return {1: [fresh, old]}.get(page, [])

        # Pretend a previous run stopped far down the history.
        from backend.app.services import sync_state

        await sync_state.record_stop(
            session, "strava", reason="throttled", detail="429",
            imported=0, listed=0, oldest_seen_on=None, resume_page=9,
        )

        count, _ = await _sync(athlete, session, _client(list_activities=AsyncMock(side_effect=_list)))
        assert count == 1
        sources = (await session.execute(select(ActivitySource))).scalars().all()
        assert {s.external_id for s in sources} == {"old", "fresh"}

    async def test_an_unresolved_source_holds_the_jump_back(self, session):
        """A source still owed its detail data is only repaired by walking past it.

        The fast-forward's one hazard, and the reason it asks first.
        """
        athlete = await _make_athlete(session, user_id="resume-3")
        base = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
        pages = {
            1: [_norm("p1", start_time=base)],
            2: [_norm("p2", start_time=base - timedelta(days=1))],
            3: [_norm("p3", start_time=base - timedelta(days=2))],
        }
        await _imported_source(session, athlete, "p1", base)
        # p2 is here but hollow — no streams were ever fetched for it.
        act = Activity(
            athlete_id=athlete.id,
            start_time=base - timedelta(days=1),
            duration_s=3600,
            status="processed",
        )
        session.add(act)
        await session.flush()
        session.add(
            ActivitySource(activity_id=act.id, provider="strava", external_id="p2")
        )
        await session.commit()

        from backend.app.services import sync_state

        await sync_state.record_stop(
            session, "strava", reason="throttled", detail="429",
            imported=0, listed=0, oldest_seen_on=None, resume_page=6,
        )

        client = _client(list_activities=AsyncMock(side_effect=lambda _t, page: pages.get(page, [])))
        await _sync(athlete, session, client)

        listed_pages = [call.args[1] for call in client.list_activities.await_args_list]
        assert listed_pages == [1, 2, 3, 4], "no jump while a source is still owed data"

    async def test_how_far_back_the_import_reached_only_ever_grows(self, session):
        """A run that stops on its first page must not erase what an earlier one reached."""
        from backend.app.services import sync_state

        athlete = await _make_athlete(session, user_id="resume-4")
        await sync_state.record_stop(
            session, "strava", reason="throttled", detail="429",
            imported=0, listed=0, oldest_seen_on=date(2012, 3, 1), resume_page=40,
        )

        await _sync(
            athlete,
            session,
            _client(
                list_activities=AsyncMock(
                    side_effect=[[_norm("act-1", start_time=datetime(2024, 6, 1, tzinfo=timezone.utc))], []]
                )
            ),
        )

        assert (await _state(session)).oldest_seen_on == date(2012, 3, 1)


class TestPacingTheBackfill:
    """A historical import is throttled so it does not hog the instance."""

    async def test_activities_are_spaced_by_the_configured_rate(self, session):
        from backend.app.core.config import settings
        from backend.app.services.provider_sync import activity_pacer

        athlete = await _make_athlete(session, user_id="pace-1")
        base = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
        norms = [_norm(f"act-{i}", start_time=base + timedelta(days=i)) for i in range(3)]

        slept: list[float] = []

        async def _record(delay):
            slept.append(delay)

        settings.sync_activities_per_minute = 600  # 0.1 s apart
        activity_pacer.reset()
        try:
            with patch("backend.app.services.provider_sync.asyncio.sleep", _record):
                count, _ = await _sync(
                    athlete,
                    session,
                    _client(list_activities=AsyncMock(side_effect=[norms, []])),
                )
        finally:
            settings.sync_activities_per_minute = 0
            activity_pacer.reset()

        assert count == 3
        # The first activity starts immediately; the two after it each wait for
        # a slot. (The waits grow rather than being 0.1 apiece because the clock
        # does not advance when the sleep itself is a mock — the count is the
        # claim here, and the interval is measured for real below.)
        assert len(slept) == 2
        assert all(delay > 0 for delay in slept)

    async def test_a_rate_of_zero_paces_nothing(self, session):
        from backend.app.core.config import settings
        from backend.app.services.provider_sync import activity_pacer

        athlete = await _make_athlete(session, user_id="pace-2")
        slept: list[float] = []

        async def _record(delay):
            slept.append(delay)

        settings.sync_activities_per_minute = 0
        activity_pacer.reset()
        with patch("backend.app.services.provider_sync.asyncio.sleep", _record):
            await _sync(
                athlete,
                session,
                _client(list_activities=AsyncMock(side_effect=[[_norm("a"), _norm("b")], []])),
            )

        assert slept == []

    async def test_an_activity_already_imported_is_not_paced(self, session):
        """A resumed run walks back through history it holds at full speed."""
        from backend.app.core.config import settings
        from backend.app.services.provider_sync import activity_pacer

        athlete = await _make_athlete(session, user_id="pace-3")
        base = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
        await _imported_source(session, athlete, "act-1", base)

        slept: list[float] = []

        async def _record(delay):
            slept.append(delay)

        settings.sync_activities_per_minute = 60
        activity_pacer.reset()
        try:
            with patch("backend.app.services.provider_sync.asyncio.sleep", _record):
                await _sync(
                    athlete,
                    session,
                    _client(
                        list_activities=AsyncMock(
                            side_effect=[[_norm("act-1", start_time=base)], []]
                        )
                    ),
                )
        finally:
            settings.sync_activities_per_minute = 0
            activity_pacer.reset()

        assert slept == []

    async def test_the_pacer_hands_out_one_slot_per_interval(self):
        """Measured against the real clock, at a rate fast enough to be cheap."""
        import time as _time

        from backend.app.core.config import settings
        from backend.app.services.provider_sync import _ActivityPacer

        pacer = _ActivityPacer()
        original = settings.sync_activities_per_minute
        settings.sync_activities_per_minute = 6000  # 10 ms apart
        try:
            start = _time.monotonic()
            for _ in range(4):
                await pacer.wait()
            elapsed = _time.monotonic() - start
        finally:
            settings.sync_activities_per_minute = original

        # Four slots: the first is free, the other three cost an interval each.
        assert elapsed >= 0.03

    async def test_two_backfills_share_one_budget(self):
        """The bound is process-wide, so concurrent imports do not each get it in full."""
        import time as _time

        from backend.app.core.config import settings
        from backend.app.services.provider_sync import activity_pacer

        original = settings.sync_activities_per_minute
        settings.sync_activities_per_minute = 6000  # 10 ms apart
        activity_pacer.reset()
        try:
            start = _time.monotonic()
            await asyncio.gather(*(activity_pacer.wait() for _ in range(4)))
            elapsed = _time.monotonic() - start
        finally:
            settings.sync_activities_per_minute = original
            activity_pacer.reset()

        assert elapsed >= 0.03


class TestBookkeepingNeverBreaksTheImport:
    """The status row exists to describe the sync, never to be able to stop it."""

    async def test_a_failed_start_marker_does_not_stop_the_import(self, session):
        from backend.app.services import sync_state

        athlete = await _make_athlete(session, user_id="bookkeeping-1")

        async def _boom(*args, **kwargs):
            raise RuntimeError("status table is unwritable")

        with patch.object(sync_state, "begin_run", _boom):
            count, _ = await _sync(
                athlete,
                session,
                _client(list_activities=AsyncMock(side_effect=[[_norm("act-1")], []])),
            )

        assert count == 1, "the ride still imports when the bookkeeping fails"

    async def test_a_failed_outcome_write_still_releases_the_lease(self, session):
        """Otherwise a status-write failure locks the next sync out for 15 minutes.

        ``record_stop`` leaves the session in pending-rollback when it fails
        mid-flush, and the lease release that follows runs on that same session.
        """
        from datetime import timedelta as _timedelta

        from backend.app.db import leases
        from backend.app.models.user_orm import SyncLease
        from backend.app.services import sync_state

        athlete = await _make_athlete(session, user_id="bookkeeping-2")

        async def _boom(failing_session, *args, **kwargs):
            # Reads first, so the failure leaves an open transaction behind it —
            # which is the state that used to strand the lease.
            await failing_session.execute(select(Activity))
            raise RuntimeError("status table is unwritable")

        with patch.object(sync_state, "record_completion", _boom):
            await _sync(
                athlete,
                session,
                _client(list_activities=AsyncMock(side_effect=[[_norm("act-1")], []])),
            )

        # The lease is free, so the next sync can take it.
        token = await leases.acquire(
            session, SyncLease, "provider-sync:strava",
            ttl=_timedelta(minutes=15), wait=0.0,
        )
        assert token is not None
        await leases.release(session, SyncLease, "provider-sync:strava", token)


class TestTheDayOfAnActivity:
    """``Activity.start_time`` is nullable, and three call sites read it.

    The guard was written out longhand at each of them before it became one
    helper; a ride with no start time must still cost nothing rather than
    stopping the walk on an ``AttributeError``.
    """

    def test_a_datetime_gives_its_date(self):
        from backend.app.services.provider_sync import _day_of

        assert _day_of(datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)) == date(2024, 6, 1)

    def test_a_date_is_already_the_answer(self):
        from backend.app.services.provider_sync import _day_of

        assert _day_of(date(2024, 6, 1)) == date(2024, 6, 1)

    def test_no_start_time_is_no_date(self):
        from backend.app.services.provider_sync import _day_of

        assert _day_of(None) is None

    async def test_an_activity_with_no_start_time_does_not_stop_the_walk(self, session):
        """The end-to-end version of the same guard."""
        athlete = await _make_athlete(session, user_id="dateless-1")
        undated = _norm("act-1")
        undated.start_time = None

        count, earliest = await _sync(
            athlete,
            session,
            _client(list_activities=AsyncMock(side_effect=[[undated], []])),
        )

        assert count == 1
        assert earliest is None
        state = await _state(session)
        assert state.status == "completed"
        assert state.oldest_seen_on is None


class TestAFailedReloadDoesNotMaskTheFailure:
    async def test_the_original_exception_still_reaches_the_caller(self, session):
        """The athlete reload on the failure path is a courtesy, not a gate.

        It runs on a session whose transaction just rolled back, so it can fail
        too — and if it did, the caller would be told about the reload instead of
        about the sync, losing the only report of what actually went wrong.
        """
        athlete = await _make_athlete(session, user_id="reload-1")
        boom = _client(list_activities=AsyncMock(side_effect=RuntimeError("boom")))

        async def _refresh_fails(*args, **kwargs):
            raise RuntimeError("the database went away too")

        with patch.object(type(session), "refresh", _refresh_fails):
            with pytest.raises(RuntimeError, match="boom"):
                await _sync(athlete, session, boom)


class TestAFailedStateReadNeverStopsTheImport:
    """The state row describes the sync; it must never be able to prevent one.

    ``begin_run`` was guarded for exactly this reason. The reads inside the walk
    run against the same new table on the same session moments later, so
    whatever makes one fail makes the others fail — and an unguarded one turns a
    bookkeeping problem into "imported nothing at all".
    """

    async def test_an_unreadable_resume_cursor_still_imports(self, session):
        from backend.app.services import sync_state

        athlete = await _make_athlete(session, user_id="state-read-1")

        async def _boom(*args, **kwargs):
            raise RuntimeError("no such table: provider_sync_states")

        with patch.object(sync_state, "resume_page_for", _boom):
            count, _ = await _sync(
                athlete,
                session,
                _client(list_activities=AsyncMock(side_effect=[[_norm("act-1")], []])),
            )

        assert count == 1, "the history still imports when the cursor cannot be read"

    async def test_an_unreadable_repair_probe_still_imports(self, session):
        """The same guarantee for the second read, which only runs with a cursor."""
        from backend.app.services import sync_state
        from backend.app.services import provider_sync as ps

        athlete = await _make_athlete(session, user_id="state-read-2")
        await sync_state.record_stop(
            session, "strava", reason="throttled", detail="429",
            imported=0, listed=10, oldest_seen_on=None, resume_page=9,
        )

        async def _boom(*args, **kwargs):
            raise RuntimeError("database is locked")

        with patch.object(ps, "_has_unresolved_source", _boom):
            count, _ = await _sync(
                athlete,
                session,
                _client(list_activities=AsyncMock(side_effect=[[_norm("act-1")], []])),
            )

        assert count == 1

    async def test_a_failed_start_marker_leaves_the_session_usable(self, session):
        """The guard must un-poison the session, or everything after it fails too."""
        from backend.app.services import sync_state

        athlete = await _make_athlete(session, user_id="state-read-3")

        async def _boom(failing_session, *args, **kwargs):
            # A *failed statement*, not a bare raise: that is what writing to a
            # missing table actually does, and it leaves the session unusable
            # for everything after it until someone rolls it back.
            from sqlalchemy import text

            await failing_session.execute(text("INSERT INTO nope VALUES (1)"))

        with patch.object(sync_state, "begin_run", _boom):
            count, _ = await _sync(
                athlete,
                session,
                _client(list_activities=AsyncMock(side_effect=[[_norm("act-1")], []])),
            )

        assert count == 1


class TestAnEarlyFailureKeepsTheCursor:
    async def test_a_run_that_listed_nothing_does_not_clobber_the_cursor(self, session):
        """Losing the cursor costs the whole point of resuming: the next run
        re-lists every page it already walked to get back to where it stopped."""
        from backend.app.services import sync_state

        athlete = await _make_athlete(session, user_id="cursor-1")
        await sync_state.record_stop(
            session, "strava", reason="throttled", detail="429",
            imported=400, listed=1000, oldest_seen_on=date(2019, 4, 2), resume_page=50,
        )

        with pytest.raises(RuntimeError):
            await _sync(
                athlete,
                session,
                _client(list_activities=AsyncMock(side_effect=RuntimeError("boom"))),
            )

        state = await _state(session)
        assert state.stop_reason == "error"
        assert state.resume_page == 50, "a run that listed nothing has nothing to say"

    async def test_a_first_stop_records_where_it_stopped(self, session):
        """With nothing recorded, any depth the walk reached is new information."""
        athlete = await _make_athlete(session, user_id="cursor-2")

        await _sync(
            athlete,
            session,
            _client(
                list_activities=AsyncMock(side_effect=[[_norm("act-1")], []]),
                get_activity_streams=AsyncMock(side_effect=_status_error(429)),
            ),
        )

        state = await _state(session)
        assert state.resume_page == 1

    async def test_a_front_sweep_failure_does_not_replace_a_deeper_cursor(self, session):
        """The ordinary way to lose the walk: a blip during the catch-up sweep.

        Every resumed run starts at the front, so a transient failure on page 1
        or 2 is the *likely* kind — and a cursor that took a dozen throttled runs
        to establish must not be replaced by the page a blip happened to reach.
        """
        from backend.app.services import sync_state

        athlete = await _make_athlete(session, user_id="cursor-3")
        base = datetime(2024, 6, 10, 10, 0, tzinfo=timezone.utc)
        await sync_state.record_stop(
            session, "strava", reason="throttled", detail="429",
            imported=400, listed=1000, oldest_seen_on=date(2019, 4, 2), resume_page=50,
        )

        async def _list(_token, page):
            if page == 1:
                return [_norm("fresh", start_time=base)]
            raise RuntimeError("the provider blew up on page 2")

        with pytest.raises(RuntimeError):
            await _sync(
                athlete, session, _client(list_activities=AsyncMock(side_effect=_list))
            )

        state = await _state(session)
        assert state.stop_reason == "error"
        assert state.resume_page == 50, "a shallow stop carries no new depth"

    async def test_a_deeper_stop_does_advance_the_cursor(self, session):
        """Advancing is the whole point — the guard must not freeze the cursor."""
        from backend.app.services import sync_state

        athlete = await _make_athlete(session, user_id="cursor-4")
        base = datetime(2024, 6, 10, 10, 0, tzinfo=timezone.utc)
        await sync_state.record_stop(
            session, "strava", reason="throttled", detail="429",
            imported=0, listed=10, oldest_seen_on=None, resume_page=3,
        )

        # Page 1 settles, the walk jumps to 2 and carries on; page 4 throws it out.
        await _imported_source(session, athlete, "p1", base)
        pages = {
            1: [_norm("p1", start_time=base)],
            2: [_norm("p2", start_time=base - timedelta(days=1))],
            3: [_norm("p3", start_time=base - timedelta(days=2))],
            4: [_norm("p4", start_time=base - timedelta(days=3))],
        }

        async def _list(_token, page):
            if page == 4:
                raise _status_error(429)
            return pages.get(page, [])

        await _sync(
            athlete,
            session,
            _client(
                list_activities=AsyncMock(side_effect=_list),
                get_activity_streams=AsyncMock(
                    return_value={"power": [200, 210, 220]}
                ),
            ),
        )

        state = await _state(session)
        assert state.resume_page == 4
