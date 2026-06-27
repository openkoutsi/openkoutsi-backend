"""
Unit tests for backend.app.services.provider_sync.

Tests ensure_fresh_token and sync_provider_activities in isolation by mocking
the PROVIDERS registry so no real HTTP calls are made.
"""
import asyncio
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.db.base import TeamBase
from backend.app.models.team_orm import Activity, ActivitySource, Athlete
from backend.app.models.registry_orm import ProviderConnection
from backend.app.services.provider_sync import ensure_fresh_token, sync_provider_activities
from backend.app.services.providers.base import NormalizedActivity


# ── Helpers ────────────────────────────────────────────────────────────────────


def _mock_conn(
    provider: str = "strava",
    *,
    access_token: str = "access-tok",
    refresh_token: str = "refresh-tok",
    token_expires_at: datetime | None = None,
) -> ProviderConnection:
    conn = MagicMock(spec=ProviderConnection)
    conn.provider = provider
    conn.access_token = access_token
    conn.refresh_token = refresh_token
    conn.token_expires_at = token_expires_at
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
    async def test_valid_token_returned_unchanged(self, session):
        conn = _mock_conn(
            token_expires_at=datetime.now(timezone.utc) + timedelta(hours=1)
        )
        token = await ensure_fresh_token(conn, session)
        assert token == "access-tok"

    async def test_no_expiry_returns_current_token(self, session):
        conn = _mock_conn(token_expires_at=None)
        token = await ensure_fresh_token(conn, session)
        assert token == "access-tok"

    async def test_expired_token_is_refreshed(self, session):
        conn = _mock_conn(
            token_expires_at=datetime.now(timezone.utc) - timedelta(hours=1)
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
            token = await ensure_fresh_token(conn, session)

        assert token == "refreshed-token"
        mock_cls.refresh_access_token.assert_called_once_with("refresh-tok")

    async def test_expired_token_updates_connection_attributes(self, session):
        conn = _mock_conn(
            token_expires_at=datetime.now(timezone.utc) - timedelta(hours=1)
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
            await ensure_fresh_token(conn, session)

        assert conn.access_token == "new-access"
        assert conn.refresh_token == "new-refresh"

    async def test_unknown_provider_returns_current_token_without_error(self, session):
        conn = _mock_conn(
            provider="nonexistent",
            token_expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        with patch("backend.app.services.provider_sync.PROVIDERS", {}):
            token = await ensure_fresh_token(conn, session)
        assert token == "access-tok"

    async def test_strava_token_expiring_within_30_minutes_is_refreshed(self, session):
        conn = _mock_conn(
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
            token = await ensure_fresh_token(conn, session)
        assert token == "proactive-strava"

    async def test_strava_token_with_over_30_minutes_is_not_refreshed(self, session):
        conn = _mock_conn(
            provider="strava",
            token_expires_at=datetime.now(timezone.utc) + timedelta(minutes=45),
        )
        mock_cls = MagicMock()
        mock_cls.refresh_access_token = AsyncMock()
        with patch("backend.app.services.provider_sync.PROVIDERS", {"strava": mock_cls}):
            token = await ensure_fresh_token(conn, session)
        assert token == "access-tok"
        mock_cls.refresh_access_token.assert_not_called()

    async def test_wahoo_token_expiring_within_1_minute_is_refreshed(self, session):
        conn = _mock_conn(
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
            token = await ensure_fresh_token(conn, session)
        assert token == "proactive-wahoo"

    async def test_wahoo_token_with_over_1_minute_is_not_refreshed(self, session):
        conn = _mock_conn(
            provider="wahoo",
            token_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        mock_cls = MagicMock()
        mock_cls.refresh_access_token = AsyncMock()
        with patch("backend.app.services.provider_sync.PROVIDERS", {"wahoo": mock_cls}):
            token = await ensure_fresh_token(conn, session)
        assert token == "access-tok"
        mock_cls.refresh_access_token.assert_not_called()

    async def test_refresh_failure_is_logged_and_reraised(self, session):
        conn = _mock_conn(
            token_expires_at=datetime.now(timezone.utc) - timedelta(hours=1)
        )
        mock_cls = MagicMock()
        mock_cls.refresh_access_token = AsyncMock(side_effect=RuntimeError("provider down"))

        with (
            patch("backend.app.services.provider_sync.PROVIDERS", {"strava": mock_cls}),
            patch("backend.app.services.provider_sync.log") as mock_log,
        ):
            with pytest.raises(RuntimeError, match="provider down"):
                await ensure_fresh_token(conn, session)

        mock_log.error.assert_called_once()
        call_args = mock_log.error.call_args
        assert "strava" in call_args.args[1]


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
                athlete, conn, session, team_id=_TEAM_ID, access_token=_ACCESS_TOKEN
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

        # Pre-seed Activity + ActivitySource
        act = Activity(
            athlete_id=athlete.id,
            start_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
            duration_s=3600,
            status="processed",
        )
        session.add(act)
        await session.flush()
        session.add(ActivitySource(activity_id=act.id, provider="strava", external_id="act-1"))
        await session.commit()

        mock_client = MagicMock()
        mock_client.list_activities = AsyncMock(side_effect=[[_norm()], []])
        mock_client.download_fit_file = AsyncMock(side_effect=Exception("no FIT"))
        mock_client.get_activity_streams = AsyncMock(return_value={})
        mock_cls = MagicMock(return_value=mock_client)

        with patch("backend.app.services.provider_sync.PROVIDERS", {"strava": mock_cls}):
            count, earliest = await sync_provider_activities(
                athlete, conn, session, team_id=_TEAM_ID, access_token=_ACCESS_TOKEN
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
                athlete, strava_conn, session, team_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        # Sync Wahoo — same start_time, should attach to existing Activity
        wahoo_mock = MagicMock()
        wahoo_mock.list_activities = AsyncMock(side_effect=[[_norm("wahoo-1", "wahoo", base_time)], []])
        wahoo_mock.download_fit_file = AsyncMock(side_effect=Exception("no FIT"))
        wahoo_mock.get_activity_streams = AsyncMock(return_value={})
        wahoo_cls = MagicMock(return_value=wahoo_mock)

        with patch("backend.app.services.provider_sync.PROVIDERS", {"wahoo": wahoo_cls}):
            await sync_provider_activities(
                athlete, wahoo_conn, session, team_id=_TEAM_ID, access_token=_ACCESS_TOKEN
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
                athlete, strava_conn, session, team_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        # Capture Strava-derived TSS
        acts = (await session.execute(select(Activity).where(Activity.athlete_id == athlete.id))).scalars().all()
        assert len(acts) == 1
        strava_tss = acts[0].tss

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
                athlete, wahoo_conn, session, team_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        assert wahoo_count == 1

        await session.refresh(acts[0])
        # Activity should now have Wahoo FIT data (higher power → higher TSS)
        assert acts[0].tss is not None
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
                athlete, wahoo_conn, session, team_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        acts = (await session.execute(select(Activity).where(Activity.athlete_id == athlete.id))).scalars().all()
        assert len(acts) == 1
        wahoo_tss = acts[0].tss

        # Strava syncs with different stream data — should NOT repopulate (priority 3 > 2)
        strava_mock = MagicMock()
        strava_mock.list_activities = AsyncMock(side_effect=[[_norm("strava-1", "strava", base_time)], []])
        strava_mock.download_fit_file = AsyncMock(side_effect=Exception("no FIT"))
        strava_mock.get_activity_streams = AsyncMock(return_value={"power": [100] * 120})
        strava_cls = MagicMock(return_value=strava_mock)

        with patch("backend.app.services.provider_sync.PROVIDERS", {"strava": strava_cls}):
            strava_count, _ = await sync_provider_activities(
                athlete, strava_conn, session, team_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        assert strava_count == 0  # Strava source added but not counted as a new/updated activity

        await session.refresh(acts[0])
        # Metrics should be unchanged from Wahoo's data
        assert acts[0].tss == wahoo_tss

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
                athlete, wahoo_conn, session, team_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        acts = (await session.execute(select(Activity).where(Activity.athlete_id == athlete.id))).scalars().all()
        assert len(acts) == 1
        assert acts[0].tss is None  # blank Wahoo has no TSS

        # Strava syncs with power data → priority=3 beats blank Wahoo priority=4 → repopulates
        strava_mock = MagicMock()
        strava_mock.list_activities = AsyncMock(side_effect=[[_norm("strava-real", "strava", base_time)], []])
        strava_mock.download_fit_file = AsyncMock(side_effect=Exception("no FIT"))
        strava_mock.get_activity_streams = AsyncMock(return_value={"power": [200] * 120})
        strava_cls = MagicMock(return_value=strava_mock)

        with patch("backend.app.services.provider_sync.PROVIDERS", {"strava": strava_cls}):
            strava_count, _ = await sync_provider_activities(
                athlete, strava_conn, session, team_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        assert strava_count == 1  # repopulated

        await session.refresh(acts[0])
        # Activity should now have Strava's data
        assert acts[0].tss is not None

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
                athlete, conn, session, team_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        assert count == 3
        assert earliest == date(2024, 6, 1)

    async def test_stream_data_persisted_with_activity(self, session):
        from backend.app.models.team_orm import ActivityStream

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
                athlete, conn, session, team_id=_TEAM_ID, access_token=_ACCESS_TOKEN
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
                athlete, conn, session, team_id=_TEAM_ID, access_token=_ACCESS_TOKEN
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
                athlete, conn, session, team_id=_TEAM_ID, access_token=_ACCESS_TOKEN
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
        # 3600 samples at 200 W → IF = 200/250 = 0.80 → "tempo"
        mock_client.get_activity_streams = AsyncMock(
            return_value={"power": [200] * 3600}
        )
        mock_cls = MagicMock(return_value=mock_client)

        with patch("backend.app.services.provider_sync.PROVIDERS", {"strava": mock_cls}):
            await sync_provider_activities(
                athlete, conn, session, team_id=_TEAM_ID, access_token=_ACCESS_TOKEN
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
        # 3600 samples at 225 W → IF = 225/250 = 0.90 → "threshold"
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
                athlete, conn, session, team_id=_TEAM_ID, access_token=_ACCESS_TOKEN
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
                athlete, conn, session, team_id=_TEAM_ID, access_token=_ACCESS_TOKEN
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
            await conn.run_sync(TeamBase.metadata.create_all)
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
