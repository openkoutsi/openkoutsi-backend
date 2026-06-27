"""
Integration tests for /api/auth/ endpoints.

Tests cover registration (invite-gated, instance-wide), login, token refresh,
logout, and account deletion against in-memory SQLite DBs. The instance is
single-tenant: there is no team slug in any path.
"""
import hashlib
import json
import secrets
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from backend.app.core.auth import hash_password
from backend.app.models.registry_orm import Invitation, User

_TEST_USER_ID = "test-user-00000000"
_PREFIX = "/api/auth"

_GOOD_PW = "Testpass1234"


# ── Helpers ────────────────────────────────────────────────────────────────────


async def _add_user(registry_session, username: str, password: str = _GOOD_PW) -> User:
    user = User(
        id=str(uuid.uuid4()),
        username=username,
        password_hash=hash_password(password),
        roles=json.dumps(["user"]),
    )
    registry_session.add(user)
    await registry_session.flush()
    return user


async def _add_invitation(
    registry_session,
    raw_token: str,
    roles: list[str] | None = None,
    expires_at: datetime | None = None,
    used_at: datetime | None = None,
) -> Invitation:
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    inv = Invitation(
        token_hash=token_hash,
        roles=json.dumps(roles or ["user"]),
        created_by_user_id=_TEST_USER_ID,
        expires_at=expires_at,
        used_at=used_at,
    )
    registry_session.add(inv)
    await registry_session.commit()
    return inv


# ── /register ─────────────────────────────────────────────────────────────────


class TestRegister:
    async def test_valid_registration_with_invite_returns_access_token(
        self, client, registry_session
    ):
        raw_token = secrets.token_hex(32)
        await _add_invitation(registry_session, raw_token)

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
        await _add_invitation(registry_session, raw_token)

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
            registry_session, raw_token,
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
            registry_session, raw_token,
            used_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        resp = await client.post(
            f"{_PREFIX}/register",
            json={"username": "useduser", "password": _GOOD_PW, "invite_token": raw_token},
        )
        assert resp.status_code == 400

    async def test_missing_invite_token_returns_422(self, client):
        resp = await client.post(
            f"{_PREFIX}/register",
            json={"username": "notoken", "password": _GOOD_PW},
        )
        assert resp.status_code == 422

    async def test_weak_password_no_uppercase_returns_422(self, client, registry_session):
        raw_token = secrets.token_hex(32)
        await _add_invitation(registry_session, raw_token)
        resp = await client.post(
            f"{_PREFIX}/register",
            json={"username": "weakuser", "password": "password1234", "invite_token": raw_token},
        )
        assert resp.status_code == 422

    async def test_weak_password_too_short_returns_422(self, client, registry_session):
        raw_token = secrets.token_hex(32)
        await _add_invitation(registry_session, raw_token)
        resp = await client.post(
            f"{_PREFIX}/register",
            json={"username": "shortuser", "password": "Short1", "invite_token": raw_token},
        )
        assert resp.status_code == 422

    async def test_weak_password_no_digit_returns_422(self, client, registry_session):
        raw_token = secrets.token_hex(32)
        await _add_invitation(registry_session, raw_token)
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
    async def test_valid_credentials_returns_access_token(self, client, registry_session):
        await _add_user(registry_session, "loginuser")
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
        await _add_user(registry_session, "realuser")
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

    async def test_deleted_account_cannot_login(self, client, registry_session):
        user = await _add_user(registry_session, "deleteduser")
        user.deleted_at = datetime.now(timezone.utc)
        await registry_session.commit()

        resp = await client.post(
            f"{_PREFIX}/login",
            json={"username": "deleteduser", "password": _GOOD_PW},
        )
        assert resp.status_code == 401


# ── /refresh ──────────────────────────────────────────────────────────────────


class TestRefresh:
    async def test_valid_cookie_returns_new_access_token(self, client, registry_session):
        await _add_user(registry_session, "refuser")
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

    async def test_access_token_as_cookie_returns_401(self, client, registry_session):
        await _add_user(registry_session, "badrefuser")
        await registry_session.commit()

        login_resp = await client.post(
            f"{_PREFIX}/login",
            json={"username": "badrefuser", "password": _GOOD_PW},
        )
        access_token = login_resp.json()["access_token"]

        client.cookies.clear()
        client.cookies.set("refresh_token", access_token, domain="test", path="/api/auth")
        resp = await client.post(f"{_PREFIX}/refresh")
        assert resp.status_code == 401

    async def test_invalid_token_string_returns_401(self, client):
        client.cookies.clear()
        client.cookies.set("refresh_token", "not.a.jwt", domain="test", path="/api/auth")
        resp = await client.post(f"{_PREFIX}/refresh")
        assert resp.status_code == 401


# ── /logout ───────────────────────────────────────────────────────────────────


class TestLogout:
    async def test_logout_returns_204(self, client):
        resp = await client.post(f"{_PREFIX}/logout")
        assert resp.status_code == 204

    async def test_logout_then_refresh_returns_401(self, client, registry_session):
        await _add_user(registry_session, "logoutuser")
        await registry_session.commit()

        await client.post(
            f"{_PREFIX}/login",
            json={"username": "logoutuser", "password": _GOOD_PW},
        )
        await client.post(f"{_PREFIX}/logout")

        resp = await client.post(f"{_PREFIX}/refresh")
        assert resp.status_code == 401


# ── /account (DELETE) ─────────────────────────────────────────────────────────


def _delete_account(client, headers=None, password=_GOOD_PW):
    """httpx.delete() doesn't accept a body; use client.request() instead."""
    h = {"Content-Type": "application/json", **(headers or {})}
    return client.request(
        "DELETE", f"{_PREFIX}/account",
        content=json.dumps({"password": password}), headers=h,
    )


class TestDeleteAccount:
    async def test_delete_account_returns_204(self, client, auth_headers):
        resp = await _delete_account(client, auth_headers)
        assert resp.status_code == 204

    async def test_wrong_password_returns_401(self, client, auth_headers):
        resp = await _delete_account(client, auth_headers, password="WrongPassword99")
        assert resp.status_code == 401

    async def test_missing_password_returns_422(self, client, auth_headers):
        h = {"Content-Type": "application/json", **auth_headers}
        resp = await client.request(
            "DELETE", f"{_PREFIX}/account", content=json.dumps({}), headers=h
        )
        assert resp.status_code == 422

    async def test_unauthenticated_delete_returns_401(self, client):
        resp = await _delete_account(client)
        assert resp.status_code == 401

    async def test_deleted_user_removed_from_registry(self, client, auth_headers, registry_session):
        from sqlalchemy import select

        resp = await _delete_account(client, auth_headers)
        assert resp.status_code == 204

        result = await registry_session.execute(
            select(User).where(User.id == _TEST_USER_ID)
        )
        assert result.scalar_one_or_none() is None

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
        with patch(
            "backend.app.api.auth.PROVIDERS",
            {"strava": AsyncMock(deauthorize=mock_deauth)},
        ):
            resp = await _delete_account(client, auth_headers)
        assert resp.status_code == 204
        mock_deauth.assert_awaited_once_with("tok")
