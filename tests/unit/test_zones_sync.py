"""
Unit tests for zone sync: provider fetch_zones implementations and the
POST /api/integrations/{provider}/sync-zones endpoint.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from backend.app.models.registry_orm import ProviderConnection
from backend.app.models.team_orm import Athlete
from backend.app.services.providers.base import ZoneData
from backend.app.services.providers.strava import StravaProviderClient, _normalize_strava_zones
from backend.app.services.providers.wahoo import _normalize_wahoo_zones

# IDs from conftest.py
_TEST_USER_ID = "test-user-00000000"
_TEST_ATHLETE_ID = "test-athlete-0000"


# ── Helpers ────────────────────────────────────────────────────────────────────


async def _add_connection(registry_session, provider: str) -> ProviderConnection:
    """Seed a ProviderConnection in the registry for the test user."""
    conn = ProviderConnection(
        user_id=_TEST_USER_ID,
        provider=provider,
        access_token="access-tok",
        refresh_token="refresh-tok",
        token_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    registry_session.add(conn)
    await registry_session.commit()
    await registry_session.refresh(conn)
    return conn


def _mock_httpx_context(responses: list) -> MagicMock:
    """Return a mock async context manager whose .get() returns responses in order."""
    mock_instance = MagicMock()
    mock_instance.get = AsyncMock(side_effect=responses)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_instance)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _mock_response(status: int, json_data: dict) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.json.return_value = json_data
    if status >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status}", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ── _normalize_strava_zones ────────────────────────────────────────────────────


class TestNormalizeStravaZones:
    def test_basic_normalization(self):
        raw = [
            {"min": 0, "max": 115},
            {"min": 115, "max": 152},
            {"min": 152, "max": 171},
            {"min": 171, "max": -1},
        ]
        result = _normalize_strava_zones(raw)
        assert len(result) == 4
        assert result[0] == {"name": "Z1", "low": 0, "high": 115}
        assert result[1] == {"name": "Z2", "low": 115, "high": 152}
        assert result[3] == {"name": "Z4", "low": 171, "high": 9999}

    def test_max_minus_one_becomes_9999(self):
        raw = [{"min": 300, "max": -1}]
        result = _normalize_strava_zones(raw)
        assert result[0]["high"] == 9999

    def test_empty_returns_empty(self):
        assert _normalize_strava_zones([]) == []


# ── _normalize_wahoo_zones ─────────────────────────────────────────────────────


class TestNormalizeWahooZones:
    def test_basic_normalization(self):
        thresholds = [144, 196, 235, 274, 313, 391, 500]
        result = _normalize_wahoo_zones(thresholds)
        assert len(result) == 7
        assert result[0] == {"name": "Z1", "low": 0, "high": 144}
        assert result[1] == {"name": "Z2", "low": 144, "high": 196}
        assert result[6] == {"name": "Z7", "low": 391, "high": 9999}

    def test_last_zone_high_is_9999(self):
        thresholds = [100, 200, 300]
        result = _normalize_wahoo_zones(thresholds)
        assert result[-1]["high"] == 9999

    def test_single_zone(self):
        result = _normalize_wahoo_zones([200])
        assert result == [{"name": "Z1", "low": 0, "high": 9999}]

    def test_empty_returns_empty(self):
        assert _normalize_wahoo_zones([]) == []


# ── StravaProviderClient.fetch_zones ──────────────────────────────────────────


class TestStravaFetchZones:
    async def test_returns_zone_data(self):
        athlete_resp = _mock_response(200, {"id": 1, "ftp": 280})
        zones_resp = _mock_response(200, {
            "heart_rate": {
                "custom_zones": False,
                "zones": [
                    {"min": 0, "max": 115},
                    {"min": 115, "max": 152},
                    {"min": 152, "max": 171},
                    {"min": 171, "max": 190},
                    {"min": 190, "max": -1},
                ],
            },
            "power": {
                "zones": [
                    {"min": 0, "max": 154},
                    {"min": 154, "max": 210},
                    {"min": 210, "max": 252},
                    {"min": 252, "max": 294},
                    {"min": 294, "max": 336},
                    {"min": 336, "max": 420},
                    {"min": 420, "max": -1},
                ]
            },
        })

        mock_ctx = _mock_httpx_context([athlete_resp, zones_resp])
        with patch("backend.app.services.providers.strava.httpx.AsyncClient", return_value=mock_ctx):
            client = StravaProviderClient()
            zone_data = await client.fetch_zones("access-tok")

        assert zone_data.ftp == 280
        assert zone_data.hr_zones is not None
        assert len(zone_data.hr_zones) == 5
        assert zone_data.hr_zones[4]["high"] == 9999
        assert zone_data.power_zones is not None
        assert len(zone_data.power_zones) == 7
        assert zone_data.power_zones[6]["high"] == 9999

    async def test_null_ftp_returns_none(self):
        athlete_resp = _mock_response(200, {"id": 1, "ftp": None})
        zones_resp = _mock_response(200, {"heart_rate": {"zones": []}, "power": {"zones": []}})
        mock_ctx = _mock_httpx_context([athlete_resp, zones_resp])
        with patch("backend.app.services.providers.strava.httpx.AsyncClient", return_value=mock_ctx):
            client = StravaProviderClient()
            zone_data = await client.fetch_zones("access-tok")
        assert zone_data.ftp is None

    async def test_empty_zones_return_none(self):
        athlete_resp = _mock_response(200, {"id": 1, "ftp": 250})
        zones_resp = _mock_response(200, {"heart_rate": {"zones": []}, "power": {"zones": []}})
        mock_ctx = _mock_httpx_context([athlete_resp, zones_resp])
        with patch("backend.app.services.providers.strava.httpx.AsyncClient", return_value=mock_ctx):
            client = StravaProviderClient()
            zone_data = await client.fetch_zones("access-tok")
        assert zone_data.hr_zones is None
        assert zone_data.power_zones is None

    async def test_raises_on_403(self):
        athlete_resp = _mock_response(403, {"message": "Authorization Error"})
        zones_resp = _mock_response(403, {"message": "Authorization Error"})
        mock_ctx = _mock_httpx_context([athlete_resp, zones_resp])
        with patch("backend.app.services.providers.strava.httpx.AsyncClient", return_value=mock_ctx):
            client = StravaProviderClient()
            with pytest.raises(httpx.HTTPStatusError):
                await client.fetch_zones("bad-token")


# ── POST /api/integrations/{provider}/sync-zones ──────────────────────────────


class TestSyncZonesEndpoint:
    async def test_sync_zones_updates_athlete(self, client, session, registry_session, auth_headers):
        await _add_connection(registry_session, "strava")

        mock_zone_data = ZoneData(
            ftp=300,
            hr_zones=[{"name": "Z1", "low": 0, "high": 120}],
            power_zones=[{"name": "Z1", "low": 0, "high": 165}],
        )

        with patch.object(StravaProviderClient, "fetch_zones", new_callable=AsyncMock, return_value=mock_zone_data):
            resp = await client.post("/api/integrations/strava/sync-zones", headers=auth_headers)

        assert resp.status_code == 200
        body = resp.json()
        assert "ftp" in body["updated"]
        assert "hr_zones" in body["updated"]
        assert "power_zones" in body["updated"]
        assert body["ftp"] == 300

        from sqlalchemy import select
        result = await session.execute(select(Athlete).where(Athlete.id == _TEST_ATHLETE_ID))
        athlete = result.scalar_one()
        assert athlete.ftp == 300
        assert athlete.hr_zones == [{"name": "Z1", "low": 0, "high": 120}]
        assert athlete.power_zones == [{"name": "Z1", "low": 0, "high": 165}]

    async def test_sync_zones_appends_ftp_history(self, client, session, registry_session, auth_headers):
        from backend.app.services.providers.wahoo import WahooClient
        await _add_connection(registry_session, "wahoo")

        mock_zone_data = ZoneData(ftp=250, power_zones=[{"name": "Z1", "low": 0, "high": 137}])

        with patch.object(WahooClient, "fetch_zones", new_callable=AsyncMock, return_value=mock_zone_data):
            resp = await client.post("/api/integrations/wahoo/sync-zones", headers=auth_headers)

        assert resp.status_code == 200

        from sqlalchemy import select
        result = await session.execute(select(Athlete).where(Athlete.id == _TEST_ATHLETE_ID))
        athlete = result.scalar_one()
        assert len(athlete.ftp_tests) == 1
        assert athlete.ftp_tests[0]["ftp"] == 250
        assert athlete.ftp_tests[0]["method"] == "wahoo"

    async def test_sync_zones_not_connected_returns_400(self, client, auth_headers):
        resp = await client.post("/api/integrations/strava/sync-zones", headers=auth_headers)
        assert resp.status_code == 400

    async def test_sync_zones_insufficient_scope_returns_403(self, client, registry_session, auth_headers):
        await _add_connection(registry_session, "strava")

        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 403

        with patch.object(
            StravaProviderClient,
            "fetch_zones",
            new_callable=AsyncMock,
            side_effect=httpx.HTTPStatusError("403 Forbidden", request=mock_request, response=mock_response),
        ):
            resp = await client.post("/api/integrations/strava/sync-zones", headers=auth_headers)

        assert resp.status_code == 403
        assert resp.json()["detail"] == "insufficient_scope"

    async def test_sync_zones_unknown_provider_returns_404(self, client, auth_headers):
        resp = await client.post("/api/integrations/garmin/sync-zones", headers=auth_headers)
        assert resp.status_code == 404
