"""Integration tests for POST /api/consent and consent visibility in /api/admin/users."""


class TestRecordConsent:
    async def test_records_consent(self, client, auth_headers):
        resp = await client.post(
            "/api/consent",
            json={"consent_version": "1.0"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["consent_version"] == "1.0"
        assert "consented_at" in data

    async def test_upserts_on_re_consent(self, client, auth_headers):
        await client.post("/api/consent", json={"consent_version": "1.0"}, headers=auth_headers)
        resp = await client.post(
            "/api/consent",
            json={"consent_version": "2.0"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["consent_version"] == "2.0"

    async def test_requires_auth(self, client):
        resp = await client.post("/api/consent", json={"consent_version": "1.0"})
        assert resp.status_code == 401

    async def test_consent_visible_in_user_list(self, client, auth_headers):
        await client.post("/api/consent", json={"consent_version": "1.0"}, headers=auth_headers)
        resp = await client.get("/api/admin/users", headers=auth_headers)
        assert resp.status_code == 200
        users = resp.json()["items"]
        assert len(users) > 0
        user = users[0]
        assert "consented_at" in user
        assert user["consented_at"] is not None
        assert user["consent_version"] == "1.0"

    async def test_user_list_shows_null_consent_when_not_given(self, client, auth_headers):
        resp = await client.get("/api/admin/users", headers=auth_headers)
        assert resp.status_code == 200
        users = resp.json()["items"]
        assert len(users) > 0
        assert users[0]["consented_at"] is None
