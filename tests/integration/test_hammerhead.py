"""
Integration tests for the Hammerhead routes (issue #69).

* ``/integrations/available`` lists Hammerhead once it is configured;
* the OAuth callback's deny redirect (``?error=access_denied``, no ``code``);
* ``POST /integrations/hammerhead/push-metrics``, including a partial grant;
* the opt-in automatic push on a profile change;
* ``POST /courses/{id}/push/hammerhead`` — create, re-push as a replace, the
  course-recon gate — and deleting the course deleting the route.
"""
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from backend.app.core.config import settings
from backend.app.models.registry_orm import ProviderConnection
from backend.app.models.user_orm import HammerheadRouteUpload
from backend.app.services.providers.hammerhead import ScopeNotGranted

_TEST_USER_ID = "test-user-00000000"
COURSE_GPX = Path(__file__).parent.parent.parent / "testdata" / "fixtures" / "synthetic_course.gpx"

_CLIENT = "backend.app.api.hammerhead.HammerheadClient"
_COURSE_CLIENT = "backend.app.api.courses.HammerheadClient"


async def _connect(registry_session) -> None:
    registry_session.add(ProviderConnection(
        user_id=_TEST_USER_ID,
        provider="hammerhead",
        provider_athlete_id="hh-1",
        access_token="hh-access",
        refresh_token="hh-refresh",
        token_expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
    ))
    await registry_session.commit()


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


# ── Metrics push ───────────────────────────────────────────────────────────


class TestPushMetrics:
    async def test_requires_a_connection(self, client, auth_headers):
        resp = await client.post(
            "/api/integrations/hammerhead/push-metrics", headers=auth_headers
        )
        assert resp.status_code == 400

    async def test_nothing_to_push(self, client, auth_headers, registry_session):
        await _connect(registry_session)
        resp = await client.post(
            "/api/integrations/hammerhead/push-metrics", headers=auth_headers
        )
        assert resp.status_code == 422

    async def test_pushes_what_is_set(
        self, client, auth_headers, registry_session, session, seeded_athlete
    ):
        await _connect(registry_session)
        seeded_athlete.ftp = 270
        seeded_athlete.weight_kg = 70.0
        await session.commit()

        with patch(f"{_CLIENT}.push_metrics", new=AsyncMock(return_value={})) as push:
            resp = await client.post(
                "/api/integrations/hammerhead/push-metrics", headers=auth_headers
            )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"pushed": ["powerFtp", "weightInKilograms"]}
        token, payload = push.await_args.args
        assert token == "hh-access"
        assert payload == {"powerFtp": 270, "weightInKilograms": 70.0}

    async def test_partial_grant_is_insufficient_scope(
        self, client, auth_headers, registry_session, session, seeded_athlete
    ):
        await _connect(registry_session)
        seeded_athlete.ftp = 270
        await session.commit()
        with patch(
            f"{_CLIENT}.push_metrics",
            new=AsyncMock(side_effect=ScopeNotGranted("metrics:write")),
        ):
            resp = await client.post(
                "/api/integrations/hammerhead/push-metrics", headers=auth_headers
            )
        assert resp.status_code == 403
        assert resp.json()["detail"] == "insufficient_scope"


class TestAutoPush:
    async def test_profile_change_pushes_when_opted_in(
        self, client, auth_headers, session, seeded_athlete
    ):
        seeded_athlete.app_settings = {"hammerhead_auto_push_metrics": True}
        await session.commit()
        with patch(
            "backend.app.api.hammerhead.auto_push_metrics", new=AsyncMock()
        ) as auto:
            resp = await client.patch(
                "/api/athlete", json={"ftp": 280}, headers=auth_headers
            )
        assert resp.status_code == 200
        auto.assert_awaited_once()

    async def test_auto_push_is_off_by_default(self, session, seeded_athlete):
        from backend.app.api.hammerhead import auto_push_metrics

        seeded_athlete.ftp = 280
        with patch("backend.app.api.hammerhead.asyncio.create_task") as create_task:
            await auto_push_metrics(_TEST_USER_ID, seeded_athlete)
        create_task.assert_not_called()

    async def test_auto_push_schedules_when_opted_in(self, session, seeded_athlete):
        from backend.app.api.hammerhead import auto_push_metrics

        seeded_athlete.ftp = 280
        seeded_athlete.app_settings = {"hammerhead_auto_push_metrics": True}
        with patch("backend.app.api.hammerhead.asyncio.create_task") as create_task:
            await auto_push_metrics(_TEST_USER_ID, seeded_athlete)
        create_task.assert_called_once()
        create_task.call_args.args[0].close()  # never awaited in this test


# ── Course push ────────────────────────────────────────────────────────────


async def _course(client, auth_headers, session, seeded_athlete) -> str:
    seeded_athlete.ftp = 250
    seeded_athlete.weight_kg = 75.0
    await session.commit()
    bike = await client.post(
        "/api/bikes",
        json={"name": "Road bike", "tyre_width_mm": 28, "riding_position": "hoods"},
        headers=auth_headers,
    )
    resp = await client.post(
        "/api/courses",
        headers=auth_headers,
        data={"bike_id": bike.json()["id"], "name": "Hill loop"},
        files={"file": (COURSE_GPX.name, COURSE_GPX.read_bytes(), "application/gpx+xml")},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


class TestCoursePush:
    @pytest.fixture(autouse=True)
    def _recon(self, course_recon_on):
        pass

    async def test_requires_a_connection(self, client, auth_headers, session, seeded_athlete):
        course_id = await _course(client, auth_headers, session, seeded_athlete)
        resp = await client.post(
            f"/api/courses/{course_id}/push/hammerhead", headers=auth_headers
        )
        assert resp.status_code == 400

    async def test_push_sends_the_original_gpx_and_re_push_replaces(
        self, client, auth_headers, session, seeded_athlete, registry_session
    ):
        await _connect(registry_session)
        course_id = await _course(client, auth_headers, session, seeded_athlete)

        upload = AsyncMock(return_value="route-1")
        with patch(f"{_COURSE_CLIENT}.upload_route", new=upload):
            first = await client.post(
                f"/api/courses/{course_id}/push/hammerhead", headers=auth_headers
            )
            second = await client.post(
                f"/api/courses/{course_id}/push/hammerhead", headers=auth_headers
            )

        assert first.json() == {"route_id": "route-1", "updated": False}
        assert second.json() == {"route_id": "route-1", "updated": True}
        first_call, second_call = upload.await_args_list
        # The exact file uploaded, decrypted — not a conversion.
        assert first_call.kwargs["gpx_bytes"] == COURSE_GPX.read_bytes()
        assert first_call.kwargs["filename"] == "Hill loop.gpx"
        assert first_call.kwargs["existing_id"] is None
        assert second_call.kwargs["existing_id"] == "route-1"

        rows = (await session.execute(select(HammerheadRouteUpload))).scalars().all()
        assert [r.hammerhead_route_id for r in rows] == ["route-1"]

        status = await client.get(
            f"/api/courses/{course_id}/push/hammerhead", headers=auth_headers
        )
        assert status.json()["route_id"] == "route-1"

    async def test_partial_grant_is_insufficient_scope(
        self, client, auth_headers, session, seeded_athlete, registry_session
    ):
        await _connect(registry_session)
        course_id = await _course(client, auth_headers, session, seeded_athlete)
        with patch(
            f"{_COURSE_CLIENT}.upload_route",
            new=AsyncMock(side_effect=ScopeNotGranted("route:write")),
        ):
            resp = await client.post(
                f"/api/courses/{course_id}/push/hammerhead", headers=auth_headers
            )
        assert resp.status_code == 403

    async def test_deleting_the_course_deletes_the_route(
        self, client, auth_headers, session, seeded_athlete, registry_session
    ):
        await _connect(registry_session)
        course_id = await _course(client, auth_headers, session, seeded_athlete)
        with patch(f"{_COURSE_CLIENT}.upload_route", new=AsyncMock(return_value="route-1")):
            await client.post(f"/api/courses/{course_id}/push/hammerhead", headers=auth_headers)

        with patch(f"{_COURSE_CLIENT}.delete_route", new=AsyncMock()) as delete_route:
            resp = await client.delete(f"/api/courses/{course_id}", headers=auth_headers)
        assert resp.status_code == 204
        delete_route.assert_awaited_once_with("hh-access", "route-1")
        assert (await session.execute(select(HammerheadRouteUpload))).scalars().all() == []

    async def test_course_deletes_even_when_the_route_delete_fails(
        self, client, auth_headers, session, seeded_athlete, registry_session
    ):
        await _connect(registry_session)
        course_id = await _course(client, auth_headers, session, seeded_athlete)
        with patch(f"{_COURSE_CLIENT}.upload_route", new=AsyncMock(return_value="route-1")):
            await client.post(f"/api/courses/{course_id}/push/hammerhead", headers=auth_headers)
        with patch(
            f"{_COURSE_CLIENT}.delete_route", new=AsyncMock(side_effect=RuntimeError("down"))
        ):
            resp = await client.delete(f"/api/courses/{course_id}", headers=auth_headers)
        assert resp.status_code == 204


class TestCoursePushGate:
    async def test_disappears_with_course_recon(self, client, auth_headers):
        resp = await client.post(
            "/api/courses/some-id/push/hammerhead", headers=auth_headers
        )
        assert resp.status_code == 404
