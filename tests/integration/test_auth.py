"""
Integration tests for /api/teams/{slug}/auth/ endpoints.

Tests cover registration (invite-gated), login, token refresh,
logout, and account deletion against in-memory SQLite DBs.
"""
import hashlib
import json
import secrets
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.core.auth import hash_password
from backend.app.models.registry_orm import Invitation, TeamMembership, User

# Matches the Team seeded in conftest.py
_SLUG = "test-team"
_TEST_TEAM_ID = "test-team-00000000"
_TEST_USER_ID = "test-user-00000000"
_PREFIX = f"/api/teams/{_SLUG}/auth"

_GOOD_PW = "Testpass1234"


# ── Helpers ────────────────────────────────────────────────────────────────────


async def _add_user(
    registry_session, username: str, password: str = _GOOD_PW
) -> User:
    user = User(
        id=str(uuid.uuid4()),
        username=username,
        password_hash=hash_password(password),
    )
    registry_session.add(user)
    await registry_session.flush()
    return user


async def _add_membership(
    registry_session, team_id: str, user_id: str, roles: list[str] | None = None
) -> TeamMembership:
    mb = TeamMembership(
        team_id=team_id,
        user_id=user_id,
        roles=json.dumps(roles or ["user"]),
    )
    registry_session.add(mb)
    await registry_session.flush()
    return mb


async def _add_invitation(
    registry_session,
    team_id: str,
    raw_token: str,
    roles: list[str] | None = None,
    expires_at: datetime | None = None,
    used_at: datetime | None = None,
) -> Invitation:
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    inv = Invitation(
        team_id=team_id,
        token_hash=token_hash,
        roles=json.dumps(roles or ["user"]),
        created_by_user_id=_TEST_USER_ID,
        expires_at=expires_at,
        used_at=used_at,
    )
    registry_session.add(inv)
    await registry_session.commit()
    return inv


@contextmanager
def _mock_team_db():
    """Mock init_team_db and get_team_session_factory for register tests."""
    from unittest.mock import MagicMock
    mock_session = AsyncMock()
    mock_session.add = MagicMock()  # add() is synchronous on SQLAlchemy sessions

    class _TeamCM:
        def __call__(self):
            return self

        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *args):
            pass

    with (
        patch("backend.app.api.auth.init_team_db", new=AsyncMock()),
        patch("backend.app.api.auth.get_team_session_factory", return_value=_TeamCM()),
    ):
        yield


# ── /register ─────────────────────────────────────────────────────────────────


class TestRegister:
    async def test_valid_registration_with_invite_returns_access_token(
        self, client, registry_session
    ):
        raw_token = secrets.token_hex(32)
        await _add_invitation(registry_session, _TEST_TEAM_ID, raw_token)

        with _mock_team_db():
            resp = await client.post(
                f"{_PREFIX}/register",
                json={"username": "newuser", "password": _GOOD_PW, "invite_token": raw_token},
            )

        assert resp.status_code == 201
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "refresh_token" not in data

    async def test_register_sets_refresh_cookie(self, client, registry_session):
        raw_token = secrets.token_hex(32)
        await _add_invitation(registry_session, _TEST_TEAM_ID, raw_token)

        with _mock_team_db():
            resp = await client.post(
                f"{_PREFIX}/register",
                json={"username": "cookieuser", "password": _GOOD_PW, "invite_token": raw_token},
            )

        assert resp.status_code == 201
        assert "refresh_token" in resp.cookies

    async def test_invalid_invite_token_returns_400(self, client):
        resp = await client.post(
            f"{_PREFIX}/register",
            json={"username": "baduser", "password": _GOOD_PW, "invite_token": "invalid-token"},
        )
        assert resp.status_code == 400

    async def test_expired_invite_returns_400(self, client, registry_session):
        raw_token = secrets.token_hex(32)
        await _add_invitation(
            registry_session,
            _TEST_TEAM_ID,
            raw_token,
            expires_at=datetime(2000, 1, 1, tzinfo=timezone.utc),
        )
        resp = await client.post(
            f"{_PREFIX}/register",
            json={"username": "expireduser", "password": _GOOD_PW, "invite_token": raw_token},
        )
        assert resp.status_code == 400

    async def test_already_used_invite_returns_400(self, client, registry_session):
        raw_token = secrets.token_hex(32)
        await _add_invitation(
            registry_session,
            _TEST_TEAM_ID,
            raw_token,
            used_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        resp = await client.post(
            f"{_PREFIX}/register",
            json={"username": "useduser", "password": _GOOD_PW, "invite_token": raw_token},
        )
        assert resp.status_code == 400

    async def test_unknown_team_slug_returns_404(self, client):
        resp = await client.post(
            "/api/teams/no-such-team/auth/register",
            json={"username": "x", "password": _GOOD_PW, "invite_token": "tok"},
        )
        assert resp.status_code == 404

    async def test_missing_invite_token_returns_422(self, client):
        resp = await client.post(
            f"{_PREFIX}/register",
            json={"username": "notoken", "password": _GOOD_PW},
        )
        assert resp.status_code == 422

    async def test_weak_password_no_uppercase_returns_422(self, client, registry_session):
        raw_token = secrets.token_hex(32)
        await _add_invitation(registry_session, _TEST_TEAM_ID, raw_token)
        resp = await client.post(
            f"{_PREFIX}/register",
            json={"username": "weakuser", "password": "password1234", "invite_token": raw_token},
        )
        assert resp.status_code == 422

    async def test_weak_password_too_short_returns_422(self, client, registry_session):
        raw_token = secrets.token_hex(32)
        await _add_invitation(registry_session, _TEST_TEAM_ID, raw_token)
        resp = await client.post(
            f"{_PREFIX}/register",
            json={"username": "shortuser", "password": "Short1", "invite_token": raw_token},
        )
        assert resp.status_code == 422

    async def test_weak_password_no_digit_returns_422(self, client, registry_session):
        raw_token = secrets.token_hex(32)
        await _add_invitation(registry_session, _TEST_TEAM_ID, raw_token)
        resp = await client.post(
            f"{_PREFIX}/register",
            json={
                "username": "nodigituser",
                "password": "Passwordwithoutdigit",
                "invite_token": raw_token,
            },
        )
        assert resp.status_code == 422


# ── /login ────────────────────────────────────────────────────────────────────


class TestLogin:
    async def test_valid_credentials_returns_access_token(
        self, client, registry_session
    ):
        user = await _add_user(registry_session, "loginuser")
        await _add_membership(registry_session, _TEST_TEAM_ID, user.id)
        await registry_session.commit()

        resp = await client.post(
            f"{_PREFIX}/login",
            json={"username": "loginuser", "password": _GOOD_PW},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in resp.cookies

    async def test_wrong_password_returns_401(self, client, registry_session):
        user = await _add_user(registry_session, "realuser")
        await _add_membership(registry_session, _TEST_TEAM_ID, user.id)
        await registry_session.commit()

        resp = await client.post(
            f"{_PREFIX}/login",
            json={"username": "realuser", "password": "wrongpassword"},
        )
        assert resp.status_code == 401

    async def test_nonexistent_user_returns_401(self, client):
        resp = await client.post(
            f"{_PREFIX}/login",
            json={"username": "ghostuser", "password": "anything"},
        )
        assert resp.status_code == 401

    async def test_not_team_member_returns_403(self, client, registry_session):
        user = await _add_user(registry_session, "nonmember")
        await registry_session.commit()

        resp = await client.post(
            f"{_PREFIX}/login",
            json={"username": "nonmember", "password": _GOOD_PW},
        )
        assert resp.status_code == 403

    async def test_deleted_account_cannot_login(self, client, registry_session):
        user = await _add_user(registry_session, "deleteduser")
        await _add_membership(registry_session, _TEST_TEAM_ID, user.id)
        user.deleted_at = datetime.now(timezone.utc)
        await registry_session.commit()

        resp = await client.post(
            f"{_PREFIX}/login",
            json={"username": "deleteduser", "password": _GOOD_PW},
        )
        assert resp.status_code == 401

    async def test_unknown_team_returns_404(self, client):
        resp = await client.post(
            "/api/teams/no-such-team/auth/login",
            json={"username": "user", "password": "pass"},
        )
        assert resp.status_code == 404


# ── /refresh ──────────────────────────────────────────────────────────────────


class TestRefresh:
    async def test_valid_cookie_returns_new_access_token(
        self, client, registry_session
    ):
        user = await _add_user(registry_session, "refuser")
        await _add_membership(registry_session, _TEST_TEAM_ID, user.id)
        await registry_session.commit()

        await client.post(
            f"{_PREFIX}/login",
            json={"username": "refuser", "password": _GOOD_PW},
        )

        resp = await client.post(f"{_PREFIX}/refresh")
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" not in data
        assert "refresh_token" in resp.cookies

    async def test_no_cookie_returns_401(self, client):
        resp = await client.post(f"{_PREFIX}/refresh")
        assert resp.status_code == 401

    async def test_access_token_as_cookie_returns_401(
        self, client, registry_session
    ):
        user = await _add_user(registry_session, "badrefuser")
        await _add_membership(registry_session, _TEST_TEAM_ID, user.id)
        await registry_session.commit()

        login_resp = await client.post(
            f"{_PREFIX}/login",
            json={"username": "badrefuser", "password": _GOOD_PW},
        )
        access_token = login_resp.json()["access_token"]

        client.cookies.clear()
        client.cookies.set(
            "refresh_token", access_token, domain="test", path=f"/api/teams/{_SLUG}/auth"
        )
        resp = await client.post(f"{_PREFIX}/refresh")
        assert resp.status_code == 401

    async def test_invalid_token_string_returns_401(self, client):
        client.cookies.clear()
        client.cookies.set(
            "refresh_token", "not.a.jwt", domain="test", path=f"/api/teams/{_SLUG}/auth"
        )
        resp = await client.post(f"{_PREFIX}/refresh")
        assert resp.status_code == 401


# ── /logout ───────────────────────────────────────────────────────────────────


class TestLogout:
    async def test_logout_returns_204(self, client):
        resp = await client.post(f"{_PREFIX}/logout")
        assert resp.status_code == 204

    async def test_logout_then_refresh_returns_401(self, client, registry_session):
        user = await _add_user(registry_session, "logoutuser")
        await _add_membership(registry_session, _TEST_TEAM_ID, user.id)
        await registry_session.commit()

        await client.post(
            f"{_PREFIX}/login",
            json={"username": "logoutuser", "password": _GOOD_PW},
        )
        await client.post(f"{_PREFIX}/logout")

        resp = await client.post(f"{_PREFIX}/refresh")
        assert resp.status_code == 401


# ── /account (DELETE) ─────────────────────────────────────────────────────────


@contextmanager
def _mock_delete_account_team_db():
    """Mock get_team_session_factory for delete account tests."""
    from unittest.mock import MagicMock
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.execute = AsyncMock(return_value=AsyncMock(scalar_one_or_none=MagicMock(return_value=None)))

    class _TeamCM:
        def __call__(self):
            return self

        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *args):
            pass

    with patch("backend.app.api.auth.get_team_session_factory", return_value=_TeamCM()):
        yield


def _delete_account(client, headers=None, password=_GOOD_PW):
    """httpx.delete() doesn't accept a body; use client.request() instead."""
    h = {"Content-Type": "application/json", **(headers or {})}
    return client.request("DELETE", f"{_PREFIX}/account", content=json.dumps({"password": password}), headers=h)


class TestDeleteAccount:
    async def test_delete_account_returns_204(self, client, auth_headers):
        with _mock_delete_account_team_db():
            resp = await _delete_account(client, auth_headers)
        assert resp.status_code == 204

    async def test_wrong_password_returns_401(self, client, auth_headers):
        with _mock_delete_account_team_db():
            resp = await _delete_account(client, auth_headers, password="WrongPassword99")
        assert resp.status_code == 401

    async def test_missing_password_returns_422(self, client, auth_headers):
        h = {"Content-Type": "application/json", **auth_headers}
        resp = await client.request("DELETE", f"{_PREFIX}/account", content=json.dumps({}), headers=h)
        assert resp.status_code == 422

    async def test_unauthenticated_delete_returns_401(self, client):
        resp = await _delete_account(client)
        assert resp.status_code == 401

    async def test_deleted_user_removed_from_registry(self, client, auth_headers, registry_session):
        from backend.app.models.registry_orm import User
        from sqlalchemy import select

        with _mock_delete_account_team_db():
            resp = await _delete_account(client, auth_headers)
        assert resp.status_code == 204

        result = await registry_session.execute(
            select(User).where(User.id == _TEST_USER_ID)
        )
        assert result.scalar_one_or_none() is None

    async def test_delete_removes_team_memberships(self, client, auth_headers, registry_session):
        from backend.app.models.registry_orm import TeamMembership
        from sqlalchemy import select

        with _mock_delete_account_team_db():
            resp = await _delete_account(client, auth_headers)
        assert resp.status_code == 204

        result = await registry_session.execute(
            select(TeamMembership).where(TeamMembership.user_id == _TEST_USER_ID)
        )
        assert result.scalars().all() == []

    async def test_delete_revokes_provider_connections(self, client, auth_headers, registry_session):
        from backend.app.models.registry_orm import ProviderConnection

        conn = ProviderConnection(
            user_id=_TEST_USER_ID,
            provider="strava",
            provider_athlete_id="12345",
            access_token="tok",
            refresh_token="ref",
            token_expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
            scopes="read",
        )
        registry_session.add(conn)
        await registry_session.commit()

        mock_deauth = AsyncMock()
        with (
            _mock_delete_account_team_db(),
            patch("backend.app.api.auth.PROVIDERS", {"strava": AsyncMock(deauthorize=mock_deauth)}),
        ):
            resp = await _delete_account(client, auth_headers)
        assert resp.status_code == 204
        mock_deauth.assert_awaited_once_with("tok")
