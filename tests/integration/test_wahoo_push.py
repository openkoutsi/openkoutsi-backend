"""
Integration tests for POST /api/workouts/{id}/push/wahoo.

Pushes a structured workout to Wahoo as a plan + scheduled workout pair. The
Wahoo HTTP calls are mocked; we assert on idempotency tracking, the visibility
window, and error handling.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import httpx
from sqlalchemy import select

from backend.app.models.registry_orm import ProviderConnection
from backend.app.models.team_orm import WahooWorkoutUpload
from backend.app.services.providers.wahoo import WahooClient

_TEST_USER_ID = "test-user-00000000"
_TEST_ATHLETE_ID = "test-athlete-0000"

_WORKOUT_BODY = {
    "name": "Threshold Session",
    "description": "2x20 min threshold",
    "sport_type": "Ride",
    "steps": [
        {"kind": "step", "step_type": "warmup", "duration": {"type": "time", "seconds": 600}},
        {"kind": "step", "step_type": "active", "duration": {"type": "time", "seconds": 1200},
         "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 90.0}}},
    ],
}


async def _add_wahoo_connection(registry_session):
    conn = ProviderConnection(
        user_id=_TEST_USER_ID,
        provider="wahoo",
        access_token="test-access-token",
        refresh_token="test-refresh-token",
        token_expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
    )
    registry_session.add(conn)
    await registry_session.commit()
    return conn


async def _create_workout(client, auth_headers) -> str:
    resp = await client.post("/api/workouts/", json=_WORKOUT_BODY, headers=auth_headers)
    assert resp.status_code == 201
    return resp.json()["id"]


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.wahooligan.com/v1/plans")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("err", request=request, response=response)


class TestPushToWahoo:
    async def test_happy_path_creates_plan_and_workout(self, client, auth_headers, registry_session, session):
        await _add_wahoo_connection(registry_session)
        workout_id = await _create_workout(client, auth_headers)

        with (
            patch.object(WahooClient, "create_or_update_plan",
                         new=AsyncMock(return_value="plan-123")),
            patch.object(WahooClient, "create_or_update_workout",
                         new=AsyncMock(return_value="workout-456")) as mock_workout,
        ):
            resp = await client.post(
                f"/api/workouts/{workout_id}/push/wahoo", json={}, headers=auth_headers
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["plan_id"] == "plan-123"
        assert body["workout_id"] == "workout-456"

        # First push: no existing Wahoo workout id passed.
        assert mock_workout.call_args.kwargs["existing_id"] is None

        rows = (await session.execute(select(WahooWorkoutUpload))).scalars().all()
        assert len(rows) == 1
        assert rows[0].wahoo_plan_id == "plan-123"
        assert rows[0].wahoo_workout_id == "workout-456"
        assert rows[0].external_id == f"okoutsi-wd-{workout_id}"

    async def test_repush_updates_in_place(self, client, auth_headers, registry_session, session):
        await _add_wahoo_connection(registry_session)
        workout_id = await _create_workout(client, auth_headers)

        with (
            patch.object(WahooClient, "create_or_update_plan",
                         new=AsyncMock(return_value="plan-123")),
            patch.object(WahooClient, "create_or_update_workout",
                         new=AsyncMock(return_value="workout-456")) as mock_workout,
        ):
            await client.post(f"/api/workouts/{workout_id}/push/wahoo", json={}, headers=auth_headers)
            resp2 = await client.post(f"/api/workouts/{workout_id}/push/wahoo", json={}, headers=auth_headers)

        assert resp2.status_code == 200
        # Second push reuses the stored Wahoo workout id (PUT, not a new record).
        assert mock_workout.call_args.kwargs["existing_id"] == "workout-456"

        rows = (await session.execute(select(WahooWorkoutUpload))).scalars().all()
        assert len(rows) == 1  # updated, not duplicated

    async def test_out_of_window_date_rejected(self, client, auth_headers, registry_session):
        await _add_wahoo_connection(registry_session)
        workout_id = await _create_workout(client, auth_headers)

        far = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
        with (
            patch.object(WahooClient, "create_or_update_plan", new=AsyncMock()),
            patch.object(WahooClient, "create_or_update_workout", new=AsyncMock()),
        ):
            resp = await client.post(
                f"/api/workouts/{workout_id}/push/wahoo",
                json={"starts": far},
                headers=auth_headers,
            )
        assert resp.status_code == 422

    async def test_past_date_rejected(self, client, auth_headers, registry_session):
        await _add_wahoo_connection(registry_session)
        workout_id = await _create_workout(client, auth_headers)

        past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        resp = await client.post(
            f"/api/workouts/{workout_id}/push/wahoo",
            json={"starts": past},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    async def test_not_connected_returns_400(self, client, auth_headers):
        workout_id = await _create_workout(client, auth_headers)
        resp = await client.post(
            f"/api/workouts/{workout_id}/push/wahoo", json={}, headers=auth_headers
        )
        assert resp.status_code == 400

    async def test_insufficient_scope_returns_403(self, client, auth_headers, registry_session):
        await _add_wahoo_connection(registry_session)
        workout_id = await _create_workout(client, auth_headers)

        with patch.object(
            WahooClient, "create_or_update_plan",
            new=AsyncMock(side_effect=_http_error(403)),
        ):
            resp = await client.post(
                f"/api/workouts/{workout_id}/push/wahoo", json={}, headers=auth_headers
            )
        assert resp.status_code == 403
        assert resp.json()["detail"] == "insufficient_scope"

    async def test_unknown_workout_returns_404(self, client, auth_headers, registry_session):
        await _add_wahoo_connection(registry_session)
        resp = await client.post(
            "/api/workouts/does-not-exist/push/wahoo", json={}, headers=auth_headers
        )
        assert resp.status_code == 404
