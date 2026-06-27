"""
Integration tests for /api/integrations endpoints.

Tests the full OAuth lifecycle (status, connect, callback, sync, disconnect)
via the HTTP test client wired to in-memory SQLite databases.

ProviderConnection records live in the registry DB (registry_session).
Activity data lives in the team DB (session).
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from jose import jwt
from sqlalchemy import select

from backend.app.core.config import settings
from backend.app.models.team_orm import Activity, ActivitySource, Athlete
from backend.app.models.registry_orm import ProviderConnection

# Fixed IDs from conftest.py
_TEST_USER_ID = "test-user-00000000"
_TEST_ATHLETE_ID = "test-athlete-0000"
_TEST_TEAM_SLUG = "test-team"


# ── Test helpers ───────────────────────────────────────────────────────────────


async def _add_connection(
    registry_session, user_id: str, provider: str
) -> ProviderConnection:
    conn = ProviderConnection(
        user_id=user_id,
        provider=provider,
        access_token="test-access-token",
        refresh_token="test-refresh-token",
        token_expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
    )
    registry_session.add(conn)
    await registry_session.commit()
    return conn


async def _add_activity(
    session, athlete_id: str, source: str, external_id: str
) -> Activity:
    act = Activity(
        athlete_id=athlete_id,
        start_time=datetime(2024, 1, 15, tzinfo=timezone.utc),
        duration_s=3600,
        status="processed",
    )
    session.add(act)
    await session.flush()
    session.add(ActivitySource(activity_id=act.id, provider=source, external_id=external_id))
    await session.commit()
    return act


def _make_session_cm(session):
    """Return a team_id → session factory mock for get_team_session_factory patches."""
    class _CM:
        def __call__(self):
            return self

        async def __aenter__(self):
            return session

        async def __aexit__(self, *args):
            pass

    return lambda team_id: _CM()


def _encode_state(user_id: str, provider: str, team_slug: str = _TEST_TEAM_SLUG) -> str:
    return jwt.encode(
        {"sub": user_id, "team_slug": team_slug, "purpose": f"{provider}_oauth"},
        settings.secret_key,
        algorithm="HS256",
    )


# ── /available ─────────────────────────────────────────────────────────────────


class TestAvailable:
    async def test_empty_when_no_providers_configured(self, client, auth_headers):
        resp = await client.get("/api/integrations/available", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == {"available": []}

    async def test_lists_strava_when_configured(self, client, auth_headers):
        with patch.object(settings, "strava_client_id", "test-strava-id"):
            resp = await client.get("/api/integrations/available", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == {"available": ["strava"]}

    async def test_lists_wahoo_when_configured(self, client, auth_headers):
        with patch.object(settings, "wahoo_client_id", "test-wahoo-id"):
            resp = await client.get("/api/integrations/available", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == {"available": ["wahoo"]}

    async def test_lists_both_when_both_configured(self, client, auth_headers):
        with (
            patch.object(settings, "strava_client_id", "test-strava-id"),
            patch.object(settings, "wahoo_client_id", "test-wahoo-id"),
        ):
            resp = await client.get("/api/integrations/available", headers=auth_headers)
        assert resp.status_code == 200
        assert set(resp.json()["available"]) == {"strava", "wahoo"}

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/integrations/available")
        assert resp.status_code == 401


# ── /status ─────────────────────────────────────────────────────────────────────


class TestStatus:
    async def test_empty_when_no_connections(self, client, auth_headers):
        resp = await client.get("/api/integrations/status", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == {"connected": []}

    async def test_lists_connected_providers(self, client, registry_session, auth_headers):
        await _add_connection(registry_session, _TEST_USER_ID, "strava")

        resp = await client.get("/api/integrations/status", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == {"connected": ["strava"]}

    async def test_lists_multiple_providers(self, client, registry_session, auth_headers):
        await _add_connection(registry_session, _TEST_USER_ID, "strava")
        await _add_connection(registry_session, _TEST_USER_ID, "wahoo")

        resp = await client.get("/api/integrations/status", headers=auth_headers)
        assert resp.status_code == 200
        assert set(resp.json()["connected"]) == {"strava", "wahoo"}

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/integrations/status")
        assert resp.status_code == 401


# ── /{provider}/connect ────────────────────────────────────────────────────────


class TestConnect:
    async def test_unknown_provider_returns_404(self, client, auth_headers):
        resp = await client.get("/api/integrations/unknown/connect", headers=auth_headers)
        assert resp.status_code == 404

    async def test_unconfigured_strava_returns_501(self, client, auth_headers):
        resp = await client.get("/api/integrations/strava/connect", headers=auth_headers)
        assert resp.status_code == 501

    async def test_configured_strava_returns_oauth_url(self, client, auth_headers):
        from backend.app.services.providers.strava import StravaProviderClient

        with (
            patch.object(settings, "strava_client_id", "test-client-id"),
            patch.object(
                StravaProviderClient,
                "get_oauth_url",
                return_value="https://strava.com/oauth/authorize?state=x",
            ),
        ):
            resp = await client.get(
                "/api/integrations/strava/connect", headers=auth_headers
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "url" in data
        assert "strava.com" in data["url"]

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/integrations/strava/connect")
        assert resp.status_code == 401


# ── /{provider}/callback ───────────────────────────────────────────────────────


class TestCallback:
    async def test_invalid_state_redirects_to_error(self, client):
        resp = await client.get(
            "/api/integrations/strava/callback?code=testcode&state=not-a-jwt",
            follow_redirects=False,
        )
        assert resp.status_code in (302, 307)
        assert "strava=error" in resp.headers["location"]

    async def test_valid_state_creates_connection_and_redirects(
        self, client, registry_session, auth_headers
    ):
        state = _encode_state(_TEST_USER_ID, "strava")

        from backend.app.services.providers.strava import StravaProviderClient

        with patch.object(
            StravaProviderClient,
            "exchange_code",
            new_callable=AsyncMock,
            return_value={
                "access_token": "new-access-token",
                "refresh_token": "new-refresh-token",
                "expires_at": 9999999999,
                "provider_athlete_id": "strava-athlete-42",
            },
        ):
            resp = await client.get(
                f"/api/integrations/strava/callback?code=authcode&state={state}",
                follow_redirects=False,
            )

        assert resp.status_code in (302, 307)
        assert "strava=connected" in resp.headers["location"]

        result = await registry_session.execute(
            select(ProviderConnection).where(
                ProviderConnection.user_id == _TEST_USER_ID,
                ProviderConnection.provider == "strava",
            )
        )
        conn = result.scalar_one_or_none()
        assert conn is not None
        assert conn.provider_athlete_id == "strava-athlete-42"

    async def test_callback_idempotent_for_existing_connection(
        self, client, registry_session, auth_headers
    ):
        """A second OAuth callback updates the existing connection instead of creating a duplicate."""
        await _add_connection(registry_session, _TEST_USER_ID, "strava")
        state = _encode_state(_TEST_USER_ID, "strava")

        from backend.app.services.providers.strava import StravaProviderClient

        with patch.object(
            StravaProviderClient,
            "exchange_code",
            new_callable=AsyncMock,
            return_value={
                "access_token": "updated-token",
                "refresh_token": "updated-refresh",
                "expires_at": 9999999999,
                "provider_athlete_id": "strava-42",
            },
        ):
            resp = await client.get(
                f"/api/integrations/strava/callback?code=code2&state={state}",
                follow_redirects=False,
            )

        assert resp.status_code in (302, 307)

        result = await registry_session.execute(
            select(ProviderConnection).where(
                ProviderConnection.user_id == _TEST_USER_ID,
                ProviderConnection.provider == "strava",
            )
        )
        connections = result.scalars().all()
        assert len(connections) == 1  # no duplicates


# ── /{provider}/sync ───────────────────────────────────────────────────────────


class TestSync:
    async def test_unknown_provider_returns_404(self, client, auth_headers):
        resp = await client.post(
            "/api/integrations/unknown/sync", headers=auth_headers
        )
        assert resp.status_code == 404

    async def test_not_connected_returns_400(self, client, auth_headers):
        resp = await client.post(
            "/api/integrations/strava/sync", headers=auth_headers
        )
        assert resp.status_code == 400

    async def test_connected_accepts_sync_request(self, client, registry_session, auth_headers):
        await _add_connection(registry_session, _TEST_USER_ID, "strava")

        resp = await client.post(
            "/api/integrations/strava/sync", headers=auth_headers
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "sync started"

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.post("/api/integrations/strava/sync")
        assert resp.status_code == 401


# ── /{provider}/disconnect ─────────────────────────────────────────────────────


class TestDisconnect:
    async def test_unauthenticated_returns_401(self, client):
        resp = await client.delete("/api/integrations/strava/disconnect")
        assert resp.status_code == 401

    async def test_not_connected_returns_400(self, client, auth_headers):
        resp = await client.delete(
            "/api/integrations/strava/disconnect", headers=auth_headers
        )
        assert resp.status_code == 400

    async def test_disconnects_removes_connection(
        self, client, registry_session, auth_headers
    ):
        await _add_connection(registry_session, _TEST_USER_ID, "strava")

        from backend.app.services.providers.strava import StravaProviderClient

        with patch.object(
            StravaProviderClient, "deauthorize", new_callable=AsyncMock
        ):
            resp = await client.delete(
                "/api/integrations/strava/disconnect", headers=auth_headers
            )

        assert resp.status_code == 204

        result = await registry_session.execute(
            select(ProviderConnection).where(
                ProviderConnection.user_id == _TEST_USER_ID,
                ProviderConnection.provider == "strava",
            )
        )
        assert result.scalar_one_or_none() is None

    async def test_keeps_activities_when_delete_data_not_set(
        self, client, session, registry_session, auth_headers
    ):
        await _add_connection(registry_session, _TEST_USER_ID, "strava")
        act = await _add_activity(session, _TEST_ATHLETE_ID, "strava", "strava-act-1")
        act_id = act.id

        from backend.app.services.providers.strava import StravaProviderClient

        with patch.object(
            StravaProviderClient, "deauthorize", new_callable=AsyncMock
        ):
            resp = await client.delete(
                "/api/integrations/strava/disconnect", headers=auth_headers
            )

        assert resp.status_code == 204

        result = await session.execute(select(Activity).where(Activity.id == act_id))
        assert result.scalar_one_or_none() is not None

    async def test_deletes_provider_activities_when_requested(
        self, client, session, registry_session, auth_headers
    ):
        await _add_connection(registry_session, _TEST_USER_ID, "strava")
        act = await _add_activity(session, _TEST_ATHLETE_ID, "strava", "strava-act-2")
        act_id = act.id

        from backend.app.services.providers.strava import StravaProviderClient

        with (
            patch.object(StravaProviderClient, "deauthorize", new_callable=AsyncMock),
            patch("backend.app.db.team_session.get_team_session_factory", _make_session_cm(session)),
        ):
            resp = await client.delete(
                "/api/integrations/strava/disconnect?delete_data=true",
                headers=auth_headers,
            )

        assert resp.status_code == 204

        session.expire_all()
        result = await session.execute(select(Activity).where(Activity.id == act_id))
        assert result.scalar_one_or_none() is None

    async def test_wahoo_disconnect_calls_deauthorize(
        self, client, registry_session, auth_headers
    ):
        await _add_connection(registry_session, _TEST_USER_ID, "wahoo")

        from backend.app.services.providers.wahoo import WahooClient

        deauth = AsyncMock()
        with patch.object(WahooClient, "deauthorize", deauth):
            resp = await client.delete(
                "/api/integrations/wahoo/disconnect", headers=auth_headers
            )

        assert resp.status_code == 204
        deauth.assert_called_once_with("test-access-token")

        result = await registry_session.execute(
            select(ProviderConnection).where(
                ProviderConnection.user_id == _TEST_USER_ID,
                ProviderConnection.provider == "wahoo",
            )
        )
        assert result.scalar_one_or_none() is None

    async def test_preserves_activities_from_other_providers(
        self, client, session, registry_session, auth_headers
    ):
        await _add_connection(registry_session, _TEST_USER_ID, "strava")
        strava_act = await _add_activity(session, _TEST_ATHLETE_ID, "strava", "strava-123")
        wahoo_act = await _add_activity(session, _TEST_ATHLETE_ID, "wahoo", "wahoo-456")
        strava_id = strava_act.id
        wahoo_id = wahoo_act.id

        from backend.app.services.providers.strava import StravaProviderClient

        with (
            patch.object(StravaProviderClient, "deauthorize", new_callable=AsyncMock),
            patch("backend.app.db.team_session.get_team_session_factory", _make_session_cm(session)),
        ):
            resp = await client.delete(
                "/api/integrations/strava/disconnect?delete_data=true",
                headers=auth_headers,
            )

        assert resp.status_code == 204

        session.expire_all()
        strava_result = await session.execute(
            select(Activity).where(Activity.id == strava_id)
        )
        wahoo_result = await session.execute(
            select(Activity).where(Activity.id == wahoo_id)
        )
        assert strava_result.scalar_one_or_none() is None
        assert wahoo_result.scalar_one_or_none() is not None
