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


class TestAllowSelfSignupSetting:
    async def test_defaults_off_and_round_trips(self, client, auth_headers):
        resp = await client.get("/api/admin/settings", headers=auth_headers)
        assert resp.json()["allow_self_signup"] is False

        resp = await client.patch(
            "/api/admin/settings",
            json={"allow_self_signup": True},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["allow_self_signup"] is True

        resp = await client.get("/api/admin/settings", headers=auth_headers)
        assert resp.json()["allow_self_signup"] is True


class TestAllowCourseReconSetting:
    """The switch an upgrading admin actually has to flip (issue #56).

    `DEPLOY.md` tells them to use this endpoint, so the endpoint is what the
    test exercises — writing the column directly would prove the ORM works and
    leave the path an admin takes untested.
    """

    async def test_defaults_off_and_round_trips(self, client, auth_headers):
        resp = await client.get("/api/admin/settings", headers=auth_headers)
        # Off unless somebody said otherwise: the half of course recon that
        # distinguishes it needs a routing sidecar the self-hoster builds tiles
        # for, and an instance that has made no decision has not said yes.
        assert resp.json()["allow_course_recon"] is False

        resp = await client.patch(
            "/api/admin/settings",
            json={"allow_course_recon": True},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["allow_course_recon"] is True

        resp = await client.get("/api/admin/settings", headers=auth_headers)
        assert resp.json()["allow_course_recon"] is True

    async def test_turning_it_on_opens_the_courses_api(self, client, auth_headers):
        """The whole point of the switch, end to end."""
        assert (await client.get("/api/courses", headers=auth_headers)).status_code == 404

        await client.patch(
            "/api/admin/settings",
            json={"allow_course_recon": True},
            headers=auth_headers,
        )
        assert (await client.get("/api/courses", headers=auth_headers)).status_code == 200

    async def test_it_is_published_to_unauthenticated_callers(self, client, auth_headers):
        """So the web app can leave Courses out of its navigation."""
        await client.patch(
            "/api/admin/settings",
            json={"allow_course_recon": True},
            headers=auth_headers,
        )
        resp = await client.get("/api/public/instance-info")
        assert resp.json()["allow_course_recon"] is True


class TestLlmModelPresetStructuredOutputs:
    async def test_defaults_on_and_round_trips_opt_out(self, client, auth_headers):
        resp = await client.patch(
            "/api/admin/settings",
            json={"llm_models": [
                {"name": "default", "base_url": "https://api.example.com/v1", "model": "m"},
                {"name": "no-schema", "base_url": "https://api.other.com/v1", "model": "m2",
                 "structured_outputs": False},
            ]},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        presets = {p["name"]: p for p in resp.json()["llm_models"]}
        # Absent flag ⇒ default-on; explicit false is preserved.
        assert presets["default"]["structured_outputs"] is True
        assert presets["no-schema"]["structured_outputs"] is False

        # And it survives a re-read.
        got = await client.get("/api/admin/settings", headers=auth_headers)
        presets = {p["name"]: p for p in got.json()["llm_models"]}
        assert presets["no-schema"]["structured_outputs"] is False


class TestPublicInstanceInfo:
    async def test_returns_null_when_unset(self, client):
        resp = await client.get("/api/public/instance-info")
        assert resp.status_code == 200
        assert resp.json()["admin_contact"] is None

    async def test_exposes_privacy_policy_url(self, client):
        resp = await client.get("/api/public/instance-info")
        assert resp.status_code == 200
        assert resp.json()["privacy_policy_url"] == "https://koutsi.dev/privacy"

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
        # Only whitelisted, non-secret fields are exposed.
        assert set(resp.json().keys()) == {
            "admin_contact",
            "privacy_policy_url",
            "email_enabled",
            "allow_self_signup",
            "allow_personal_access_tokens",
            "allow_course_recon",
        }
