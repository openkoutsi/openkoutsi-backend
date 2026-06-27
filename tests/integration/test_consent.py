"""Integration tests for POST /api/teams/{slug}/consent."""
import pytest


_SLUG = "test-team"
_PREFIX = f"/api/teams/{_SLUG}"


class TestRecordConsent:
    async def test_records_consent_for_team_member(self, client, auth_headers):
        resp = await client.post(
            f"{_PREFIX}/consent",
            json={"consent_version": "1.0"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["consent_version"] == "1.0"
        assert "consented_at" in data

    async def test_upserts_on_re_consent(self, client, auth_headers):
        await client.post(f"{_PREFIX}/consent", json={"consent_version": "1.0"}, headers=auth_headers)
        resp = await client.post(
            f"{_PREFIX}/consent",
            json={"consent_version": "2.0"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["consent_version"] == "2.0"

    async def test_requires_auth(self, client):
        resp = await client.post(f"{_PREFIX}/consent", json={"consent_version": "1.0"})
        assert resp.status_code == 401

    async def test_consent_visible_in_member_list(self, client, auth_headers):
        await client.post(f"{_PREFIX}/consent", json={"consent_version": "1.0"}, headers=auth_headers)
        resp = await client.get(f"{_PREFIX}/members", headers=auth_headers)
        assert resp.status_code == 200
        members = resp.json()
        assert len(members) > 0
        member = members[0]
        assert "consented_at" in member
        assert member["consented_at"] is not None
        assert member["consent_version"] == "1.0"

    async def test_member_list_shows_null_consent_when_not_given(self, client, auth_headers):
        resp = await client.get(f"{_PREFIX}/members", headers=auth_headers)
        assert resp.status_code == 200
        members = resp.json()
        assert len(members) > 0
        assert members[0]["consented_at"] is None
