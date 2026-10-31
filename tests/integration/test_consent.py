"""Integration tests for POST /api/consent and consent visibility in /api/admin/users."""


class TestRecordConsent:
    async def test_records_consent(self, client, auth_headers):
        resp = await client.post(
            "/api/consent",
            json={"consent_version": "1.0", "confirmed_adult": True},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["consent_version"] == "1.0"
        assert "consented_at" in data

    async def test_upserts_on_re_consent(self, client, auth_headers):
        await client.post("/api/consent", json={"consent_version": "1.0", "confirmed_adult": True}, headers=auth_headers)
        resp = await client.post(
            "/api/consent",
            json={"consent_version": "2.0", "confirmed_adult": True},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["consent_version"] == "2.0"

    async def test_refused_without_adult_confirmation(self, client, auth_headers, registry_session):
        await _clear_consent(registry_session)
        resp = await client.post(
            "/api/consent", json={"consent_version": "1.1"}, headers=auth_headers
        )
        assert resp.status_code == 422
        await _assert_not_consented(client, auth_headers)

    async def test_refused_when_adult_confirmation_false(
        self, client, auth_headers, registry_session
    ):
        await _clear_consent(registry_session)
        resp = await client.post(
            "/api/consent",
            json={"consent_version": "1.1", "confirmed_adult": False},
            headers=auth_headers,
        )
        assert resp.status_code == 422
        await _assert_not_consented(client, auth_headers)

    async def test_current_version_with_adult_confirmation_unlocks(
        self, client, auth_headers, registry_session
    ):
        from backend.app.api.consent import CURRENT_CONSENT_VERSION

        await _clear_consent(registry_session)
        resp = await client.post(
            "/api/consent",
            json={"consent_version": CURRENT_CONSENT_VERSION, "confirmed_adult": True},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        athlete = (await client.get("/api/athlete", headers=auth_headers)).json()
        assert athlete["consent_accepted"] is True

    async def test_pre_adult_check_consent_no_longer_counts(
        self, client, auth_headers, registry_session
    ):
        # Accounts that accepted 1.0 never claimed to be adults, so they are
        # sent back through the consent step.
        await _clear_consent(registry_session, version="1.0")
        await _assert_not_consented(client, auth_headers)

    async def test_requires_auth(self, client):
        resp = await client.post("/api/consent", json={"consent_version": "1.0", "confirmed_adult": True})
        assert resp.status_code == 401

    async def test_consent_visible_in_user_list(self, client, auth_headers):
        await client.post("/api/consent", json={"consent_version": "1.0", "confirmed_adult": True}, headers=auth_headers)
        resp = await client.get("/api/admin/users", headers=auth_headers)
        assert resp.status_code == 200
        users = resp.json()["items"]
        assert len(users) > 0
        user = users[0]
        assert "consented_at" in user
        assert user["consented_at"] is not None
        assert user["consent_version"] == "1.0"

    async def test_user_list_shows_null_consent_when_not_given(
        self, client, auth_headers, registry_session
    ):
        await _clear_consent(registry_session)
        resp = await client.get("/api/admin/users", headers=auth_headers)
        assert resp.status_code == 200
        users = resp.json()["items"]
        assert len(users) > 0
        assert users[0]["consented_at"] is None


async def _clear_consent(registry_session, version=None):
    """Reset the seeded user to the un-consented state (default is consented).

    With ``version``, leave a consent on record for that (outdated) version.
    """
    from datetime import datetime, timezone

    from sqlalchemy import select

    from backend.app.models.registry_orm import User

    user = (await registry_session.execute(select(User))).scalar_one()
    user.consented_at = datetime.now(timezone.utc) if version else None
    user.consent_version = version
    await registry_session.commit()


async def _assert_not_consented(client, auth_headers):
    athlete = (await client.get("/api/athlete", headers=auth_headers)).json()
    assert athlete["consent_accepted"] is False
    resp = await client.get("/api/integrations/strava/connect", headers=auth_headers)
    assert resp.status_code == 403


class TestConsentEnforcement:
    """The API — not just the UI — refuses ingestion without current consent."""

    async def test_upload_blocked_without_consent(self, client, auth_headers, registry_session):
        await _clear_consent(registry_session)
        resp = await client.post(
            "/api/activities/upload",
            headers=auth_headers,
            files={"file": ("ride.fit", b"not-a-real-fit", "application/octet-stream")},
        )
        assert resp.status_code == 403

    async def test_connect_blocked_without_consent(self, client, auth_headers, registry_session):
        await _clear_consent(registry_session)
        resp = await client.get("/api/integrations/strava/connect", headers=auth_headers)
        assert resp.status_code == 403

    async def test_connect_passes_gate_with_consent(self, client, auth_headers):
        # Seeded user is consented → the consent gate does not block (any
        # non-403 outcome, e.g. 200 or 501 if the provider is unconfigured).
        resp = await client.get("/api/integrations/strava/connect", headers=auth_headers)
        assert resp.status_code != 403
