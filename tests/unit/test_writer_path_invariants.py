"""Every path that populates an activity from streams must set the aerobic metrics.

Four paths do it — ``process_fit_file``, the reprocess endpoint, and both
provider-sync paths (FIT download and the stream-based Strava fallback). Issue
#37 originally wired only the first two, so a provider-synced ride came back
with a null ``decoupling_pct`` *and* a null ``decoupling_reason``, a combination
``Activity`` documents as impossible.

The fix was to wire the other two. This is the guard that stops a fifth path
from silently reintroducing the same hole: it asserts the model's stated
invariant — exactly one of the two is set — after each writer runs.
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from backend.app.models.user_orm import Activity, ActivityStream, Athlete
from backend.app.services.provider_sync import sync_provider_activities
from backend.app.services.providers.base import NormalizedActivity

_TEAM_ID = "test-team"
_ACCESS_TOKEN = "access-tok"
_START = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
_DURATION_S = 4000


def _steady_power() -> list[float]:
    """A long, evenly paced ride — the shape that *should* yield a figure."""
    return [200.0 + (i % 7) for i in range(_DURATION_S)]


def _steady_hr() -> list[float]:
    return [140.0 + (i % 5) for i in range(_DURATION_S)]


async def _make_athlete(session, user_id: str) -> Athlete:
    athlete = Athlete(global_user_id=user_id, ftp=250, ftp_tests=[])
    session.add(athlete)
    await session.commit()
    await session.refresh(athlete)
    return athlete


def _make_connection(athlete: Athlete, provider: str) -> MagicMock:
    conn = MagicMock()
    conn.user_id = athlete.global_user_id
    conn.provider = provider
    conn.access_token = _ACCESS_TOKEN
    conn.refresh_token = "refresh-tok"
    conn.token_expires_at = None
    return conn


def _norm(provider: str) -> NormalizedActivity:
    return NormalizedActivity(
        external_id=f"{provider}-1",
        source=provider,
        name="Test Ride",
        sport_type="Ride",
        start_time=_START,
        duration_s=_DURATION_S,
        distance_m=50_000.0,
        elevation_m=500.0,
        avg_power=None,
        avg_hr=None,
        max_hr=None,
        avg_speed_ms=14.0,
        avg_cadence=None,
    )


def _assert_invariant(activity: Activity) -> None:
    """Exactly one of decoupling_pct / decoupling_reason, as `Activity` states."""
    has_value = activity.decoupling_pct is not None
    has_reason = activity.decoupling_reason is not None
    assert has_value != has_reason, (
        "activity has "
        f"decoupling_pct={activity.decoupling_pct!r} and "
        f"decoupling_reason={activity.decoupling_reason!r} — exactly one must be set. "
        "A writer path is not calling apply_aerobic_metrics."
    )


class TestProviderSyncWriterPaths:
    async def test_stream_based_fallback_sets_aerobic_metrics(self, session):
        """Strava path: no FIT to download, metrics come from the API streams."""
        athlete = await _make_athlete(session, "user-strava")
        conn = _make_connection(athlete, "strava")

        client = MagicMock()
        client.list_activities = AsyncMock(side_effect=[[_norm("strava")], []])
        client.download_fit_file = AsyncMock(side_effect=Exception("no FIT"))
        client.get_activity_streams = AsyncMock(return_value={
            "power": _steady_power(),
            "heartrate": _steady_hr(),
            "cadence": [90.0] * _DURATION_S,
            "speed": [10.0] * _DURATION_S,
            "altitude": [100.0] * _DURATION_S,
        })

        with patch(
            "backend.app.services.provider_sync.PROVIDERS",
            {"strava": MagicMock(return_value=client)},
        ):
            await sync_provider_activities(
                athlete, conn, session, user_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        activity = (
            await session.execute(select(Activity).where(Activity.athlete_id == athlete.id))
        ).scalar_one()
        _assert_invariant(activity)
        # A long evenly-paced ride with both streams qualifies, so this is the
        # positive case: a real figure, not merely a well-formed refusal.
        assert activity.decoupling_pct is not None

    async def test_fit_download_path_sets_aerobic_metrics(self, session):
        """Wahoo/Garmin path: metrics come from the downloaded FIT."""
        athlete = await _make_athlete(session, "user-wahoo")
        conn = _make_connection(athlete, "wahoo")

        profile = MagicMock()
        profile.power = _steady_power()
        profile.heartRate = _steady_hr()
        profile.cadence = [90.0] * _DURATION_S
        profile.speed = [36.0] * _DURATION_S
        profile.altitude = [100.0] * _DURATION_S
        profile.avgHeartRate = 142.0
        profile.peakHR = 160.0
        profile.avgPower = 203.0
        profile.avgCadence = 90
        profile.avgSpeed = 36.0
        profile.duration = _DURATION_S
        profile.distance = 50_000
        profile.elevationGain = 500
        profile.start_time = _START
        profile.sport_type = "cycling"

        client = MagicMock()
        client.list_activities = AsyncMock(side_effect=[[_norm("wahoo")], []])
        client.download_fit_file = AsyncMock(return_value=b".FITfake")
        client.get_activity_streams = AsyncMock(return_value={})

        with (
            patch(
                "backend.app.services.provider_sync.PROVIDERS",
                {"wahoo": MagicMock(return_value=client)},
            ),
            patch("backend.app.services.provider_sync.summarizeWorkout", return_value=profile),
            patch("backend.app.services.provider_sync.extractIntervals", return_value=[]),
            patch("backend.app.services.provider_sync.encrypt_file"),
        ):
            await sync_provider_activities(
                athlete, conn, session, user_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        activity = (
            await session.execute(select(Activity).where(Activity.athlete_id == athlete.id))
        ).scalar_one()
        _assert_invariant(activity)
        assert activity.decoupling_pct is not None

    async def test_synced_ride_without_heart_rate_gets_a_reason_not_a_null(self, session):
        """The invariant holds on the negative case too — a reason, never both null."""
        athlete = await _make_athlete(session, "user-nohr")
        conn = _make_connection(athlete, "strava")

        client = MagicMock()
        client.list_activities = AsyncMock(side_effect=[[_norm("strava")], []])
        client.download_fit_file = AsyncMock(side_effect=Exception("no FIT"))
        client.get_activity_streams = AsyncMock(return_value={
            "power": _steady_power(),
            "heartrate": [],
            "cadence": [],
            "speed": [10.0] * _DURATION_S,
            "altitude": [],
        })

        with patch(
            "backend.app.services.provider_sync.PROVIDERS",
            {"strava": MagicMock(return_value=client)},
        ):
            await sync_provider_activities(
                athlete, conn, session, user_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        activity = (
            await session.execute(select(Activity).where(Activity.athlete_id == athlete.id))
        ).scalar_one()
        _assert_invariant(activity)
        assert activity.decoupling_reason == "no_hr"

    @pytest.mark.parametrize("provider", ["strava", "wahoo"])
    async def test_w_bal_stream_absent_without_a_cp_fit(self, session, provider):
        """A first-ever sync has no power history, so no CP — and no invented W'."""
        athlete = await _make_athlete(session, f"user-nocp-{provider}")
        conn = _make_connection(athlete, provider)

        client = MagicMock()
        client.list_activities = AsyncMock(side_effect=[[_norm(provider)], []])
        client.download_fit_file = AsyncMock(side_effect=Exception("no FIT"))
        client.get_activity_streams = AsyncMock(return_value={
            "power": [200.0] * 60,  # too short to produce any CP-fit duration best
            "heartrate": [140.0] * 60,
            "cadence": [],
            "speed": [],
            "altitude": [],
        })

        with patch(
            "backend.app.services.provider_sync.PROVIDERS",
            {provider: MagicMock(return_value=client)},
        ):
            await sync_provider_activities(
                athlete, conn, session, user_id=_TEAM_ID, access_token=_ACCESS_TOKEN
            )

        activity = (
            await session.execute(select(Activity).where(Activity.athlete_id == athlete.id))
        ).scalar_one()
        _assert_invariant(activity)
        assert activity.cp_w is None
        assert activity.w_prime_j is None

        streams = (
            await session.execute(
                select(ActivityStream).where(ActivityStream.activity_id == activity.id)
            )
        ).scalars().all()
        assert "w_bal" not in {s.stream_type for s in streams}
