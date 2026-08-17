"""A long password must be a 422, not a 500 (issue #102, F-07).

bcrypt hashes at most 72 bytes and, since 5.0, raises rather than truncating:
``ValueError: password cannot be longer than 72 bytes``. The strength validator
enforced a *minimum* length, an uppercase letter and a digit — but no maximum,
and nothing caught the exception:

    POST /api/auth/login, 103-byte password  -> HTTP 500   (expected 401)

On login that is an unauthenticated 500 generator, and #108 (F-06) widened it:
verifying unconditionally means any identifier now reaches bcrypt, not just a
known one. On signup it meant someone with a long passphrase could not create an
account and got a server error rather than an explanation.

Every password field is covered here, including the two that only *check* a
password (login, delete-account) and the setup wizard's, which restated the
strength rules in its own copy and so never gained the maximum.

The limit is on the UTF-8 encoding, not the character count, so the boundary
cases below are expressed in bytes.
"""
import json
import uuid

import pytest

from backend.app.core.auth import hash_password
from backend.app.models.registry_orm import User

_PREFIX = "/api/auth"
_GOOD_PW = "Testpass1234"

# 72 bytes exactly — the largest bcrypt accepts, and still strong enough to
# pass the other rules (uppercase, digit, >= 12 characters).
_AT_LIMIT = "A1" + "a" * 70
# One byte over.
_OVER_LIMIT = "A1" + "a" * 71
# 36 two-byte characters = 72 bytes: at the limit while only half as long in
# characters, which is the case a naive `len(v) > 72` check gets wrong.
_AT_LIMIT_MULTIBYTE = "A1" + "é" * 35
# 40 two-byte characters = 80 bytes, but only 42 characters.
_OVER_LIMIT_MULTIBYTE = "A1" + "é" * 39


def _assert_byte_lengths():
    assert len(_AT_LIMIT.encode()) == 72
    assert len(_OVER_LIMIT.encode()) == 73
    assert len(_AT_LIMIT_MULTIBYTE.encode()) == 72
    assert len(_OVER_LIMIT_MULTIBYTE.encode()) == 80


_assert_byte_lengths()


async def _add_user(registry_session, username: str = "known-user") -> User:
    user = User(
        id=str(uuid.uuid4()),
        username=username,
        password_hash=hash_password(_GOOD_PW),
        roles=json.dumps(["user"]),
    )
    registry_session.add(user)
    await registry_session.commit()
    return user


class TestLogin:
    """The unauthenticated one."""

    async def test_over_limit_is_422_not_500(self, client, registry_session):
        await _add_user(registry_session)
        resp = await client.post(
            f"{_PREFIX}/login", json={"username": "known-user", "password": _OVER_LIMIT}
        )
        assert resp.status_code == 422

    async def test_over_limit_is_422_for_unknown_user_too(self, client):
        """F-06 made this reachable without a valid identifier."""
        resp = await client.post(
            f"{_PREFIX}/login", json={"username": "nobody", "password": _OVER_LIMIT}
        )
        assert resp.status_code == 422

    async def test_multibyte_over_limit_is_rejected(self, client):
        """80 bytes in 42 characters — a character-count check would let it through."""
        resp = await client.post(
            f"{_PREFIX}/login",
            json={"username": "nobody", "password": _OVER_LIMIT_MULTIBYTE},
        )
        assert resp.status_code == 422

    async def test_at_limit_reaches_authentication(self, client, registry_session):
        """72 bytes is valid input: it gets a 401, not a 422 and not a 500."""
        await _add_user(registry_session)
        resp = await client.post(
            f"{_PREFIX}/login", json={"username": "known-user", "password": _AT_LIMIT}
        )
        assert resp.status_code == 401

    async def test_multibyte_at_limit_reaches_authentication(
        self, client, registry_session
    ):
        await _add_user(registry_session)
        resp = await client.post(
            f"{_PREFIX}/login",
            json={"username": "known-user", "password": _AT_LIMIT_MULTIBYTE},
        )
        assert resp.status_code == 401

    async def test_the_message_says_what_is_wrong(self, client):
        """A user with a long passphrase needs to know why, in bytes."""
        resp = await client.post(
            f"{_PREFIX}/login", json={"username": "nobody", "password": _OVER_LIMIT}
        )
        detail = json.dumps(resp.json()).lower()
        assert "72" in detail and "byte" in detail

    async def test_login_still_works_at_a_normal_length(self, client, registry_session):
        await _add_user(registry_session)
        resp = await client.post(
            f"{_PREFIX}/login", json={"username": "known-user", "password": _GOOD_PW}
        )
        assert resp.status_code == 200


class TestPasswordSettingEndpoints:
    """Every route that hashes a new password."""

    async def test_signup_rejects_over_limit(self, client):
        resp = await client.post(
            f"{_PREFIX}/signup",
            json={"email": "long@example.com", "password": _OVER_LIMIT},
        )
        assert resp.status_code == 422

    async def test_register_rejects_over_limit(self, client):
        resp = await client.post(
            f"{_PREFIX}/register",
            json={
                "username": "newbie",
                "password": _OVER_LIMIT,
                "invite_token": "irrelevant",
            },
        )
        assert resp.status_code == 422

    async def test_reset_password_rejects_over_limit(self, client):
        resp = await client.post(
            f"{_PREFIX}/reset-password",
            json={"token": "irrelevant", "new_password": _OVER_LIMIT},
        )
        assert resp.status_code == 422

    async def test_setup_rejects_over_limit(self, client):
        """The setup wizard kept its own copy of the rules and so missed this."""
        resp = await client.post(
            "/api/setup",
            json={"admin_username": "admin", "admin_password": _OVER_LIMIT},
        )
        assert resp.status_code == 422

    @pytest.mark.parametrize("password", [_AT_LIMIT, _AT_LIMIT_MULTIBYTE])
    async def test_setting_at_the_limit_is_allowed(self, client, password):
        """72 bytes must still be usable — the fix is a ceiling, not a haircut.

        A 400 here means the request got past validation and was refused on its
        merits (no invite token), which is the proof wanted: not a 422.
        """
        resp = await client.post(
            f"{_PREFIX}/register",
            json={
                "username": "newbie",
                "password": password,
                "invite_token": "not-a-real-token",
            },
        )
        assert resp.status_code == 400


class TestDeleteAccount:
    async def test_rejects_over_limit(self, client, auth_headers):
        resp = await client.request(
            "DELETE",
            f"{_PREFIX}/account",
            json={"password": _OVER_LIMIT},
            headers=auth_headers,
        )
        assert resp.status_code == 422


class TestStrengthRulesUnchanged:
    """The maximum must not have displaced the rules that were already there."""

    @pytest.mark.parametrize("password,missing", [
        ("Short1", "12 characters"),
        ("alllowercase123", "uppercase"),
        ("NoDigitsAtAllHere", "digit"),
    ])
    async def test_weak_passwords_still_rejected(self, client, password, missing):
        resp = await client.post(
            f"{_PREFIX}/signup", json={"email": "weak@example.com", "password": password}
        )
        assert resp.status_code == 422
        assert missing in json.dumps(resp.json())

    async def test_login_does_not_apply_the_strength_rules(
        self, client, registry_session
    ):
        """Login checks length only.

        Applying the full policy to a *login* would lecture someone about
        uppercase letters instead of rejecting their attempt, and would lock
        out any account whose password predates the current rules.
        """
        user = await _add_user(registry_session, username="legacy")
        user.password_hash = hash_password("weak")
        await registry_session.commit()

        resp = await client.post(
            f"{_PREFIX}/login", json={"username": "legacy", "password": "weak"}
        )
        assert resp.status_code == 200
