"""
Integration tests for /api/setup/ and /api/teams/{slug}/ endpoints.

Covers:
- Setup status and first-run creation
- Member listing, role updates, removal, password-reset link generation
- Invitation creation, listing, and revocation
- Team settings (LLM config) get and update
- Role enforcement (403 for non-admin/coach callers)
- Coach member-access endpoints
"""
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.core.auth import TeamContext, create_access_token, hash_password
from backend.app.db.base import RegistryBase
from backend.app.db.registry import get_registry_session
from backend.app.models.registry_orm import Invitation, Team, TeamMembership, User
from backend.main import create_app

_SLUG = "test-team"
_TEST_TEAM_ID = "test-team-00000000"
_TEST_USER_ID = "test-user-00000000"
_PREFIX = f"/api/teams/{_SLUG}"


def _make_headers(roles: list[str]) -> dict:
    """JWT auth headers for the seeded test user with specific roles."""
    token = create_access_token(_TEST_USER_ID, _TEST_TEAM_ID, roles)
    return {"Authorization": f"Bearer {token}"}


def _admin_headers() -> dict:
    return _make_headers(["administrator", "user"])


def _user_headers() -> dict:
    return _make_headers(["user"])


def _coach_headers() -> dict:
    return _make_headers(["coach", "user"])


# ── Setup fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
async def empty_registry_engine():
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with eng.begin() as conn:
        await conn.run_sync(RegistryBase.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
async def empty_registry_session(empty_registry_engine):
    factory = async_sessionmaker(empty_registry_engine, expire_on_commit=False)
    async with factory() as s:
        yield s


@pytest.fixture
async def setup_client(empty_registry_session):
    """Client with empty registry (no teams) for setup endpoint tests."""
    app = create_app()

    async def _override():
        yield empty_registry_session

    app.dependency_overrides[get_registry_session] = _override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


def _mock_team_db_ctx():
    """Context manager that patches init_team_db and get_team_session_factory."""
    mock_session = AsyncMock()
    mock_session.add = MagicMock()

    class _CM:
        def __call__(self):
            return self

        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *args):
            pass

    return patch("backend.app.api.setup.init_team_db", new=AsyncMock()), patch(
        "backend.app.api.setup.get_team_session_factory", return_value=_CM()
    )


# ── /api/setup/ ────────────────────────────────────────────────────────────────


class TestSetupStatus:
    async def test_empty_registry_returns_needs_setup_true(self, setup_client):
        resp = await setup_client.get("/api/setup/status")
        assert resp.status_code == 200
        assert resp.json()["needs_setup"] is True

    async def test_populated_registry_returns_needs_setup_false(self, client):
        resp = await client.get("/api/setup/status")
        assert resp.status_code == 200
        assert resp.json()["needs_setup"] is False


class TestFirstRunSetup:
    async def test_creates_team_and_returns_access_token(self, setup_client):
        p1, p2 = _mock_team_db_ctx()
        with p1, p2:
            resp = await setup_client.post(
                "/api/setup",
                json={
                    "team_name": "My Club",
                    "slug": "my-club",
                    "admin_username": "admin",
                    "admin_password": "Adminpass1234",
                },
            )
        assert resp.status_code == 201
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    async def test_second_call_returns_409(self, setup_client):
        p1, p2 = _mock_team_db_ctx()
        with p1, p2:
            await setup_client.post(
                "/api/setup",
                json={
                    "team_name": "First",
                    "slug": "first",
                    "admin_username": "admin",
                    "admin_password": "Adminpass1234",
                },
            )
            resp = await setup_client.post(
                "/api/setup",
                json={
                    "team_name": "Second",
                    "slug": "second",
                    "admin_username": "admin2",
                    "admin_password": "Adminpass1234",
                },
            )
        assert resp.status_code == 409


# ── /api/teams/{slug}/members ──────────────────────────────────────────────────


class TestListMembers:
    async def test_admin_can_list_members(self, client, registry_session):
        resp = await client.get(f"{_PREFIX}/members", headers=_admin_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert any(m["user_id"] == _TEST_USER_ID for m in data)

    async def test_coach_can_list_members(self, client, registry_session):
        resp = await client.get(f"{_PREFIX}/members", headers=_coach_headers())
        assert resp.status_code == 200

    async def test_plain_user_gets_403(self, client):
        resp = await client.get(f"{_PREFIX}/members", headers=_user_headers())
        assert resp.status_code == 403

    async def test_unauthenticated_gets_401(self, client):
        resp = await client.get(f"{_PREFIX}/members")
        assert resp.status_code == 401


class TestUpdateMemberRoles:
    async def test_admin_can_update_roles(self, client, registry_session):
        resp = await client.patch(
            f"{_PREFIX}/members/{_TEST_USER_ID}/roles",
            headers=_admin_headers(),
            json={"roles": ["user"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["roles"] == ["user"]

    async def test_invalid_role_returns_400(self, client, registry_session):
        resp = await client.patch(
            f"{_PREFIX}/members/{_TEST_USER_ID}/roles",
            headers=_admin_headers(),
            json={"roles": ["superuser"]},
        )
        assert resp.status_code == 400

    async def test_nonexistent_member_returns_404(self, client, registry_session):
        resp = await client.patch(
            f"{_PREFIX}/members/{uuid.uuid4()}/roles",
            headers=_admin_headers(),
            json={"roles": ["user"]},
        )
        assert resp.status_code == 404

    async def test_plain_user_gets_403(self, client):
        resp = await client.patch(
            f"{_PREFIX}/members/{_TEST_USER_ID}/roles",
            headers=_user_headers(),
            json={"roles": ["user"]},
        )
        assert resp.status_code == 403


class TestRemoveMember:
    async def test_admin_can_remove_member(self, client, registry_session):
        other_user = User(
            id=str(uuid.uuid4()),
            username="to-remove",
            password_hash=hash_password("Testpass1234"),
        )
        registry_session.add(other_user)
        await registry_session.flush()
        registry_session.add(
            TeamMembership(
                team_id=_TEST_TEAM_ID,
                user_id=other_user.id,
                roles='["user"]',
            )
        )
        await registry_session.commit()

        resp = await client.delete(
            f"{_PREFIX}/members/{other_user.id}",
            headers=_admin_headers(),
        )
        assert resp.status_code == 204

    async def test_nonexistent_member_returns_404(self, client, registry_session):
        resp = await client.delete(
            f"{_PREFIX}/members/{uuid.uuid4()}",
            headers=_admin_headers(),
        )
        assert resp.status_code == 404

    async def test_plain_user_gets_403(self, client):
        resp = await client.delete(
            f"{_PREFIX}/members/{_TEST_USER_ID}",
            headers=_user_headers(),
        )
        assert resp.status_code == 403


class TestPasswordResetLink:
    async def test_admin_gets_reset_url(self, client, registry_session):
        resp = await client.post(
            f"{_PREFIX}/members/{_TEST_USER_ID}/password-reset",
            headers=_admin_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "reset_url" in data
        assert "/reset-password" in data["reset_url"]

    async def test_nonexistent_member_returns_404(self, client, registry_session):
        resp = await client.post(
            f"{_PREFIX}/members/{uuid.uuid4()}/password-reset",
            headers=_admin_headers(),
        )
        assert resp.status_code == 404

    async def test_plain_user_gets_403(self, client):
        resp = await client.post(
            f"{_PREFIX}/members/{_TEST_USER_ID}/password-reset",
            headers=_user_headers(),
        )
        assert resp.status_code == 403


# ── /api/teams/{slug}/invitations ─────────────────────────────────────────────


class TestCreateInvitation:
    async def test_admin_can_create_invitation(self, client, registry_session):
        resp = await client.post(
            f"{_PREFIX}/invitations",
            headers=_admin_headers(),
            json={"roles": ["user"], "expires_in_days": 7},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "url" in data
        assert "/register?token=" in data["url"]
        assert data["roles"] == ["user"]

    async def test_invitation_with_no_expiry(self, client, registry_session):
        resp = await client.post(
            f"{_PREFIX}/invitations",
            headers=_admin_headers(),
            json={"roles": ["user"], "expires_in_days": None},
        )
        assert resp.status_code == 201
        assert resp.json()["expires_at"] is None

    async def test_invalid_role_returns_400(self, client, registry_session):
        resp = await client.post(
            f"{_PREFIX}/invitations",
            headers=_admin_headers(),
            json={"roles": ["superadmin"]},
        )
        assert resp.status_code == 400

    async def test_plain_user_gets_403(self, client):
        resp = await client.post(
            f"{_PREFIX}/invitations",
            headers=_user_headers(),
            json={"roles": ["user"]},
        )
        assert resp.status_code == 403


class TestListInvitations:
    async def test_admin_can_list_invitations(self, client, registry_session):
        # Create one invitation first
        await client.post(
            f"{_PREFIX}/invitations",
            headers=_admin_headers(),
            json={"roles": ["user"]},
        )
        resp = await client.get(f"{_PREFIX}/invitations", headers=_admin_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    async def test_plain_user_gets_403(self, client):
        resp = await client.get(f"{_PREFIX}/invitations", headers=_user_headers())
        assert resp.status_code == 403


class TestRevokeInvitation:
    async def test_admin_can_revoke_pending_invitation(self, client, registry_session):
        create_resp = await client.post(
            f"{_PREFIX}/invitations",
            headers=_admin_headers(),
            json={"roles": ["user"]},
        )
        invitation_id = create_resp.json()["id"]

        resp = await client.delete(
            f"{_PREFIX}/invitations/{invitation_id}",
            headers=_admin_headers(),
        )
        assert resp.status_code == 204

    async def test_cannot_revoke_used_invitation(self, client, registry_session):
        from datetime import datetime, timezone

        inv = Invitation(
            id=str(uuid.uuid4()),
            team_id=_TEST_TEAM_ID,
            token_hash="deadbeef" * 8,
            roles='["user"]',
            created_by_user_id=_TEST_USER_ID,
            used_at=datetime.now(timezone.utc),
            used_by_user_id=_TEST_USER_ID,
        )
        registry_session.add(inv)
        await registry_session.commit()

        resp = await client.delete(
            f"{_PREFIX}/invitations/{inv.id}",
            headers=_admin_headers(),
        )
        assert resp.status_code == 400

    async def test_nonexistent_invitation_returns_404(self, client, registry_session):
        resp = await client.delete(
            f"{_PREFIX}/invitations/{uuid.uuid4()}",
            headers=_admin_headers(),
        )
        assert resp.status_code == 404

    async def test_plain_user_gets_403(self, client):
        resp = await client.delete(
            f"{_PREFIX}/invitations/{uuid.uuid4()}",
            headers=_user_headers(),
        )
        assert resp.status_code == 403


# ── /api/teams/{slug}/settings ────────────────────────────────────────────────


class TestTeamSettings:
    async def test_admin_can_get_settings(self, client, registry_session):
        resp = await client.get(f"{_PREFIX}/settings", headers=_admin_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert "llm_base_url" in data
        assert "llm_model" in data
        assert "llm_api_key_set" in data

    async def test_plain_user_get_settings_403(self, client):
        resp = await client.get(f"{_PREFIX}/settings", headers=_user_headers())
        assert resp.status_code == 403

    async def test_admin_can_update_llm_settings(self, client, registry_session):
        resp = await client.patch(
            f"{_PREFIX}/settings",
            headers=_admin_headers(),
            json={"llm_base_url": "https://api.openai.com/v1", "llm_model": "gpt-4o"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["llm_base_url"] == "https://api.openai.com/v1"
        assert data["llm_model"] == "gpt-4o"

    async def test_clear_llm_base_url(self, client, registry_session):
        # Set then clear
        await client.patch(
            f"{_PREFIX}/settings",
            headers=_admin_headers(),
            json={"llm_base_url": "https://example.com/v1"},
        )
        resp = await client.patch(
            f"{_PREFIX}/settings",
            headers=_admin_headers(),
            json={"llm_base_url": ""},
        )
        assert resp.status_code == 200
        assert resp.json()["llm_base_url"] is None

    async def test_plain_user_update_settings_403(self, client):
        resp = await client.patch(
            f"{_PREFIX}/settings",
            headers=_user_headers(),
            json={"llm_model": "gpt-4o"},
        )
        assert resp.status_code == 403


# ── /api/members/{member_user_id}/ (coach access) ────────────────────────────


class TestCoachMemberAccess:
    async def test_admin_can_get_member_athlete(self, client, registry_session, auth_headers):
        resp = await client.get(
            f"/api/members/{_TEST_USER_ID}/athlete", headers=auth_headers
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "global_user_id" in data or "id" in data

    async def test_admin_can_list_member_activities(self, client, registry_session, auth_headers):
        resp = await client.get(
            f"/api/members/{_TEST_USER_ID}/activities", headers=auth_headers
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data

    async def test_plain_user_cannot_access_member_athlete(self, client, registry_session):
        from backend.app.core.deps import get_ctx_and_session
        from fastapi import Request, HTTPException

        # Override ctx_session to return user (non-admin, non-coach) context
        test_ctx = TeamContext(user_id=_TEST_USER_ID, team_id=_TEST_TEAM_ID, roles=["user"])

        # We need to temporarily replace the override so that ctx has "user" role
        # The client fixture overrides get_ctx_and_session; we patch it for this test
        app = create_app()

        async def _user_ctx_session(request: Request):
            if not request.headers.get("Authorization", "").startswith("Bearer "):
                raise HTTPException(status_code=401)
            from backend.app.db.team_session import get_team_session_factory
            # yield a user-only context - no real session needed, will 403 before DB access
            yield test_ctx, None

        from backend.app.db.registry import get_registry_session as _reg

        async def _override_registry():
            yield registry_session

        app.dependency_overrides[get_ctx_and_session] = _user_ctx_session
        app.dependency_overrides[_reg] = _override_registry

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get(
                f"/api/members/{_TEST_USER_ID}/athlete",
                headers=_user_headers(),
            )
        assert resp.status_code == 403

    async def test_nonexistent_member_returns_404(self, client, registry_session, auth_headers):
        resp = await client.get(
            f"/api/members/{uuid.uuid4()}/athlete", headers=auth_headers
        )
        assert resp.status_code == 404
