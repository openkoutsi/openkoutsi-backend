"""
Integration tests for Hammerhead through the generic integration routes (issue #69).

* ``/integrations/available`` lists Hammerhead once it is configured;
* ``/connect`` answers 501 unconfigured, and the authorize URL otherwise;
* the OAuth callback's deny redirect (``?error=access_denied``, no ``code``).
"""
from unittest.mock import patch

from backend.app.core.config import settings


# ── Generic integration routes ─────────────────────────────────────────────


class TestGenericRoutes:
    async def test_available_lists_hammerhead_when_configured(self, client, auth_headers):
        with patch.object(settings, "hammerhead_client_id", "hh-id"):
            resp = await client.get("/api/integrations/available", headers=auth_headers)
        assert resp.json() == {"available": ["hammerhead"]}

    async def test_connect_is_501_when_unconfigured(self, client, auth_headers):
        resp = await client.get("/api/integrations/hammerhead/connect", headers=auth_headers)
        assert resp.status_code == 501

    async def test_connect_returns_the_authorize_url(self, client, auth_headers):
        with patch.object(settings, "hammerhead_client_id", "hh-id"):
            resp = await client.get(
                "/api/integrations/hammerhead/connect", headers=auth_headers
            )
        assert resp.status_code == 200
        assert resp.json()["url"].startswith(
            "https://api.hammerhead.io/v1/auth/oauth/authorize?"
        )

    async def test_deny_redirect_is_reported_as_denied(self, client):
        resp = await client.get(
            "/api/integrations/hammerhead/callback?error=access_denied&state=whatever",
            follow_redirects=False,
        )
        assert resp.status_code in (302, 307)
        assert resp.headers["location"].endswith("/profile?hammerhead=denied")

    async def test_callback_without_code_is_an_error(self, client):
        resp = await client.get(
            "/api/integrations/hammerhead/callback?state=whatever",
            follow_redirects=False,
        )
        assert resp.headers["location"].endswith("/profile?hammerhead=error")
