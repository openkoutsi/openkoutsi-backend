"""Integration tests for the admin_contact instance setting.

Covers admin read/write via /api/admin/settings and the unauthenticated
GET /api/public/instance-info endpoint.
"""


class TestAdminContactAdmin:
    async def test_patch_sets_admin_contact(self, client, auth_headers):
        resp = await client.patch(
            "/api/admin/settings",
            json={"admin_contact": "admin@example.com"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["admin_contact"] == "admin@example.com"

    async def test_get_echoes_admin_contact(self, client, auth_headers):
        await client.patch(
            "/api/admin/settings",
            json={"admin_contact": "reach-me@example.com"},
            headers=auth_headers,
        )
        resp = await client.get("/api/admin/settings", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["admin_contact"] == "reach-me@example.com"

    async def test_empty_string_clears_admin_contact(self, client, auth_headers):
        await client.patch(
            "/api/admin/settings",
            json={"admin_contact": "admin@example.com"},
            headers=auth_headers,
        )
        resp = await client.patch(
            "/api/admin/settings",
            json={"admin_contact": ""},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["admin_contact"] is None

    async def test_admin_settings_requires_auth(self, client):
        resp = await client.get("/api/admin/settings")
        assert resp.status_code == 401


class TestPublicInstanceInfo:
    async def test_returns_null_when_unset(self, client):
        resp = await client.get("/api/public/instance-info")
        assert resp.status_code == 200
        assert resp.json() == {"admin_contact": None}

    async def test_returns_configured_value_without_auth(self, client, auth_headers):
        await client.patch(
            "/api/admin/settings",
            json={"admin_contact": "admin@example.com"},
            headers=auth_headers,
        )
        # No auth headers on this call — the endpoint is public.
        resp = await client.get("/api/public/instance-info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["admin_contact"] == "admin@example.com"

    async def test_exposes_only_admin_contact(self, client, auth_headers):
        # Configure the full instance settings to ensure nothing else leaks.
        await client.patch(
            "/api/admin/settings",
            json={
                "admin_contact": "admin@example.com",
                "llm_base_url": "https://llm.example.com",
                "llm_model": "some-model",
                "llm_analysis_context": "secret context",
            },
            headers=auth_headers,
        )
        resp = await client.get("/api/public/instance-info")
        assert resp.status_code == 200
        assert set(resp.json().keys()) == {"admin_contact"}
