"""
Unit tests for backend.app.services.wahoo_sync.process_wahoo_webhook.
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from backend.app.models.registry_orm import ProviderConnection, TeamMembership
from backend.app.models.team_orm import Activity, ActivitySource, Athlete
from backend.app.services.wahoo_sync import process_wahoo_webhook

# Payload matching the real-world Wahoo webhook structure observed in production:
# workout is nested inside workout_summary, not at the top level.
WAHOO_PAYLOAD = {
    "event_type": "workout_summary",
    "webhook_token": "sdf098sd0f8s9d8f08sdf",
    "user": {
        "id": 9876543
    },
    "workout_summary": {
        "id": 12341234,
        "started_at": "2026-04-25T09:36:48.000Z",
        "ascent_accum": "103.0",
        "cadence_avg": "75.0",
        "calories_accum": "806.0",
        "distance_accum": "27441.58",
        "duration_active_accum": "4184.0",
        "duration_paused_accum": "337.0",
        "duration_total_accum": "4521.0",
        "heart_rate_avg": "160.0",
        "power_bike_np_last": "218.0",
        "power_bike_tss_last": "95.3",
        "power_avg": "191.0",
        "speed_avg": "6.56",
        "work_accum": "797068.0",
        "fitness_app_id": 1,
        "time_zone": "Europe/Helsinki",
        "created_at": "2026-04-25T10:52:37.000Z",
        "updated_at": "2026-04-25T10:52:38.000Z",
        "file": {
            "url": "https://example.com/fit_files/myworkout.fit"
        },
        "workout": {
            "id": 1234567890,
            "starts": "2026-04-25T09:36:48.000Z",
            "minutes": 75,
            "name": "Gravel cycling",
            "created_at": "2026-04-25T10:52:37.000Z",
            "updated_at": "2026-04-25T10:52:37.000Z",
            "plan_id": None,
            "workout_token": "ELEMNT BOLT AABB:CC",
            "workout_type_id": 0,
            "fitness_app_id": 1,
        },
    },
}

_WAHOO_USER_ID = "9876543"
_GLOBAL_USER_ID = "test-user-webhook-1"
_TEAM_ID = "test-team-webhook"


def _make_session_cm(session):
    """Return a callable that acts as an async context manager yielding session."""
    class _CM:
        def __call__(self):
            return self

        async def __aenter__(self):
            return session

        async def __aexit__(self, *args):
            pass

    return _CM()


async def _seed_athlete_and_conn(team_session, registry_session):
    athlete = Athlete(global_user_id=_GLOBAL_USER_ID, ftp_tests=[])
    team_session.add(athlete)
    await team_session.flush()

    # SQLite does not enforce FK constraints by default, so we can insert
    # ProviderConnection and TeamMembership without creating User/Team rows.
    conn = ProviderConnection(
        user_id=_GLOBAL_USER_ID,
        provider="wahoo",
        provider_athlete_id=_WAHOO_USER_ID,
        access_token="access-tok",
        refresh_token="refresh-tok",
    )
    registry_session.add(conn)

    membership = TeamMembership(
        team_id=_TEAM_ID,
        user_id=_GLOBAL_USER_ID,
        roles='["user"]',
    )
    registry_session.add(membership)
    await registry_session.flush()

    return athlete, conn


@pytest.mark.asyncio
async def test_process_wahoo_webhook_nested_workout_creates_activity(session, registry_session):
    """
    Wahoo posts workout nested inside workout_summary (not at top level).
    process_wahoo_webhook must extract it, normalise it, and create an Activity.
    """
    await _seed_athlete_and_conn(session, registry_session)

    with (
        patch(
            "backend.app.db.registry._RegistrySessionLocal",
            new=_make_session_cm(registry_session),
        ),
        patch(
            "backend.app.db.team_session.get_team_session_factory",
            return_value=_make_session_cm(session),
        ),
        patch(
            "backend.app.services.wahoo_sync.ensure_fresh_token",
            new=AsyncMock(return_value="access-tok"),
        ),
        patch(
            "backend.app.services.wahoo_sync._wahoo_client.download_fit_file",
            new=AsyncMock(side_effect=Exception("no FIT in test")),
        ),
        patch(
            "backend.app.services.provider_sync._populate_activity",
            new=AsyncMock(),
        ),
        patch(
            "backend.app.services.metrics_engine.recalculate_from",
            new=AsyncMock(),
        ),
    ):
        await process_wahoo_webhook(WAHOO_PAYLOAD)

    activities = (await session.execute(select(Activity))).scalars().all()
    assert len(activities) == 1, "Expected exactly one Activity to be created"

    act = activities[0]
    assert act.name == "Gravel cycling"
    assert act.sport_type == "Ride"  # workout_type_id=0 maps to Ride
    # SQLite drops timezone info; compare as naive UTC
    assert act.start_time.replace(tzinfo=None) == datetime(2026, 4, 25, 9, 36, 48)
    assert act.distance_m == pytest.approx(27441.58)
    assert act.elevation_m == pytest.approx(103.0)
    assert act.avg_power == pytest.approx(191.0)
    assert act.avg_hr == pytest.approx(160.0)
    assert act.avg_cadence == pytest.approx(75.0)

    sources = (await session.execute(select(ActivitySource))).scalars().all()
    assert len(sources) == 1
    assert sources[0].provider == "wahoo"
    assert sources[0].external_id == "1234567890"


@pytest.mark.asyncio
async def test_process_wahoo_webhook_missing_user_ignored(session, registry_session):
    """Payloads without user.id must be silently ignored."""
    payload = {**WAHOO_PAYLOAD, "user": {}}
    # Function returns early — no session factories needed
    await process_wahoo_webhook(payload)

    activities = (await session.execute(select(Activity))).scalars().all()
    assert activities == []


@pytest.mark.asyncio
async def test_wahoo_deauthorize_sends_delete_to_permissions_endpoint():
    """WahooClient.deauthorize must DELETE /v1/permissions with the Bearer token."""
    from unittest.mock import MagicMock

    from backend.app.services.providers.wahoo import WahooClient

    mock_client = AsyncMock()
    mock_client.delete = AsyncMock(return_value=MagicMock())
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        await WahooClient.deauthorize("my-test-token")

    mock_client.delete.assert_called_once_with(
        "https://api.wahooligan.com/v1/permissions",
        headers={"Authorization": "Bearer my-test-token"},
    )


@pytest.mark.asyncio
async def test_process_wahoo_webhook_missing_workout_ignored(session, registry_session):
    """Payloads without any workout object (neither top-level nor nested) must be ignored."""
    summary_without_workout = {
        k: v for k, v in WAHOO_PAYLOAD["workout_summary"].items() if k != "workout"
    }
    payload = {**WAHOO_PAYLOAD, "workout_summary": summary_without_workout}
    payload.pop("workout", None)

    # Function returns early after detecting missing workout — no registry calls
    await process_wahoo_webhook(payload)

    activities = (await session.execute(select(Activity))).scalars().all()
    assert activities == []
