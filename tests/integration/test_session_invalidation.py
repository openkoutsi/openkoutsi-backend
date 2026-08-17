"""Integration tests for session invalidation (issue #102, F-04).

The reproduction in the finding, verbatim:

    logged in: access token + refresh cookie obtained
    password reset -> HTTP 204
    OLD access token still works?               HTTP 200  YES
    OLD refresh cookie still mints new tokens?  HTTP 200  YES

The reset handler already revoked every personal access token, with a comment
explaining that whatever prompted the reset applies to the credentials the
account handed out. Sessions were untouched — and there was no mechanism to
touch them with, since the JWTs carry only ``sub``, ``exp`` and ``type``.

``User.token_version`` is that mechanism: stamped into both token types as
``ver``, compared on every request and on refresh, bumped by a reset and by an
explicit sign-out-everywhere.

Like ``test_personal_access_tokens.py``, this module builds its own client
overriding only the registry session. The shared ``client`` fixture in
``conftest.py`` overrides ``get_ctx_and_session`` and waves every ``Bearer …``
through, which would make every assertion here pass without the fix present.
"""
import hashlib
import json
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.core.auth import (
    claimed_token_version,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
)
from backend.app.core.encryption import set_user_encryption_context
from backend.app.db.registry import get_registry_session
from backend.app.db.user_session import get_user_session_factory, init_user_db
from backend.app.models.registry_orm import PasswordResetToken, User
from backend.app.models.user_orm import Athlete

_PREFIX = "/api/auth"
_GOOD_PW = "Testpass1234"
_NEW_PW = "Newpass12345"
_COOKIE = "refresh_token"


@pytest.fixture
async def auth_client(app, registry_session):
    """A client with the **real** identity path — only the registry is overridden.

    Never request this and the shared ``client`` fixture in the same test: both
    write ``dependency_overrides`` on the module-scoped app, and ``client``'s
    override accepts any bearer value.
    """
    from backend.app.core.limiter import limiter

    async def _override_registry():
        yield registry_session

    app.dependency_overrides[get_registry_session] = _override_registry
    limiter.enabled = False
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c
    finally:
        limiter.enabled = True
        app.dependency_overrides.clear()


async def _make_user(registry_session, *, username="reset-me") -> User:
    """A real account: registry row, own database, athlete profile.

    The per-user DB is provisioned because the real identity path opens it —
    an authenticated request that reached a user with no database would fail
    for that reason rather than the one under test.
    """
    from backend.app.api.consent import CURRENT_CONSENT_VERSION

    user = User(
        id=str(uuid.uuid4()),
        username=username,
        password_hash=hash_password(_GOOD_PW),
        roles=json.dumps(["user"]),
        consented_at=datetime.now(timezone.utc),
        consent_version=CURRENT_CONSENT_VERSION,
    )
    registry_session.add(user)
    await registry_session.commit()

    set_user_encryption_context(user.id)
    await init_user_db(user.id)
    async with get_user_session_factory(user.id)() as s:
        s.add(Athlete(id=str(uuid.uuid4()), global_user_id=user.id, name="T", ftp_tests=[]))
        await s.commit()
    return user


async def _login(auth_client, username: str, password: str = _GOOD_PW):
    """Return (access_token, refresh_cookie) for a fresh login."""
    resp = await auth_client.post(
        f"{_PREFIX}/login", json={"username": username, "password": password}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"], resp.cookies.get(_COOKIE)


async def _issue_reset_token(registry_session, user: User) -> str:
    raw = secrets.token_urlsafe(32)
    registry_session.add(
        PasswordResetToken(
            id=str(uuid.uuid4()),
            user_id=user.id,
            token_hash=hashlib.sha256(raw.encode()).hexdigest(),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
    )
    await registry_session.commit()
    return raw


class TestPasswordResetEndsSessions:
    async def test_old_access_token_stops_working(self, auth_client, registry_session):
        user = await _make_user(registry_session)
        access, _ = await _login(auth_client, "reset-me")

        assert (await auth_client.get(
            "/api/athlete", headers={"Authorization": f"Bearer {access}"}
        )).status_code != 401

        raw = await _issue_reset_token(registry_session, user)
        resp = await auth_client.post(
            f"{_PREFIX}/reset-password", json={"token": raw, "new_password": _NEW_PW}
        )
        assert resp.status_code == 204

        after = await auth_client.get(
            "/api/athlete", headers={"Authorization": f"Bearer {access}"}
        )
        assert after.status_code == 401

    async def test_old_refresh_cookie_stops_minting_tokens(self, auth_client, registry_session):
        """The 30-day half. Refusing only the access token would buy an hour."""
        user = await _make_user(registry_session)
        _, refresh_cookie = await _login(auth_client, "reset-me")

        assert (await auth_client.post(
            f"{_PREFIX}/refresh", cookies={_COOKIE: refresh_cookie}
        )).status_code == 200

        raw = await _issue_reset_token(registry_session, user)
        await auth_client.post(
            f"{_PREFIX}/reset-password", json={"token": raw, "new_password": _NEW_PW}
        )

        after = await auth_client.post(
            f"{_PREFIX}/refresh", cookies={_COOKIE: refresh_cookie}
        )
        assert after.status_code == 401

    async def test_new_login_after_reset_works(self, auth_client, registry_session):
        """Invalidation must not lock the legitimate owner out."""
        user = await _make_user(registry_session)
        await _login(auth_client, "reset-me")

        raw = await _issue_reset_token(registry_session, user)
        await auth_client.post(
            f"{_PREFIX}/reset-password", json={"token": raw, "new_password": _NEW_PW}
        )

        access, refresh_cookie = await _login(auth_client, "reset-me", _NEW_PW)
        assert (await auth_client.get(
            "/api/athlete", headers={"Authorization": f"Bearer {access}"}
        )).status_code != 401
        assert (await auth_client.post(
            f"{_PREFIX}/refresh", cookies={_COOKIE: refresh_cookie}
        )).status_code == 200

    async def test_reset_bumps_the_version_once(self, auth_client, registry_session):
        user = await _make_user(registry_session)
        assert user.token_version == 0

        raw = await _issue_reset_token(registry_session, user)
        await auth_client.post(
            f"{_PREFIX}/reset-password", json={"token": raw, "new_password": _NEW_PW}
        )

        await registry_session.refresh(user)
        assert user.token_version == 1

    async def test_another_users_sessions_are_untouched(self, auth_client, registry_session):
        """Bumping one account must not sign the instance out."""
        victim = await _make_user(registry_session, username="reset-me")
        await _make_user(registry_session, username="bystander")

        bystander_access, bystander_cookie = await _login(auth_client, "bystander")

        raw = await _issue_reset_token(registry_session, victim)
        await auth_client.post(
            f"{_PREFIX}/reset-password", json={"token": raw, "new_password": _NEW_PW}
        )

        assert (await auth_client.get(
            "/api/athlete", headers={"Authorization": f"Bearer {bystander_access}"}
        )).status_code != 401
        assert (await auth_client.post(
            f"{_PREFIX}/refresh", cookies={_COOKIE: bystander_cookie}
        )).status_code == 200


class TestLogoutAll:
    async def test_ends_every_session_including_the_caller(self, auth_client, registry_session):
        await _make_user(registry_session, username="everywhere")
        first_access, first_cookie = await _login(auth_client, "everywhere")
        second_access, second_cookie = await _login(auth_client, "everywhere")

        resp = await auth_client.post(
            f"{_PREFIX}/logout-all", headers={"Authorization": f"Bearer {second_access}"}
        )
        assert resp.status_code == 204

        for token in (first_access, second_access):
            assert (await auth_client.get(
                "/api/athlete", headers={"Authorization": f"Bearer {token}"}
            )).status_code == 401
        for cookie in (first_cookie, second_cookie):
            assert (await auth_client.post(
                f"{_PREFIX}/refresh", cookies={_COOKIE: cookie}
            )).status_code == 401

    async def test_requires_authentication(self, auth_client):
        assert (await auth_client.post(f"{_PREFIX}/logout-all")).status_code == 401

    async def test_signing_in_again_works(self, auth_client, registry_session):
        await _make_user(registry_session, username="everywhere")
        access, _ = await _login(auth_client, "everywhere")
        await auth_client.post(
            f"{_PREFIX}/logout-all", headers={"Authorization": f"Bearer {access}"}
        )

        fresh, _ = await _login(auth_client, "everywhere")
        assert (await auth_client.get(
            "/api/athlete", headers={"Authorization": f"Bearer {fresh}"}
        )).status_code != 401

    async def test_plain_logout_leaves_other_sessions_alone(self, auth_client, registry_session):
        """Documented boundary: /logout is this browser, /logout-all is the account."""
        await _make_user(registry_session, username="everywhere")
        other_access, _ = await _login(auth_client, "everywhere")
        mine_access, _ = await _login(auth_client, "everywhere")

        resp = await auth_client.post(
            f"{_PREFIX}/logout", headers={"Authorization": f"Bearer {mine_access}"}
        )
        assert resp.status_code == 204
        assert (await auth_client.get(
            "/api/athlete", headers={"Authorization": f"Bearer {other_access}"}
        )).status_code != 401


class TestTokensCarryTheVersion:
    def test_access_token_stamps_the_version(self):
        token = create_access_token("u1", ["user"], token_version=7)
        assert decode_token(token)["ver"] == 7

    def test_refresh_token_stamps_the_version(self):
        token = create_refresh_token("u1", token_version=7)
        assert decode_token(token)["ver"] == 7

    def test_token_minted_before_the_column_reads_as_zero(self):
        """The upgrade must not sign the instance out.

        Existing users land on token_version 0, and a pre-upgrade token has no
        ``ver`` claim at all, so it keeps working until it expires. Only this
        instance's key signs these, so an absent claim cannot be chosen by a
        caller — the first reset after the upgrade moves the user to 1 and takes
        the old token with it.
        """
        assert claimed_token_version({"sub": "u1", "type": "access"}) == 0

    async def test_pre_upgrade_token_still_authenticates(self, auth_client, registry_session):
        """The same property, end to end: a token minted with no ``ver`` claim."""
        from jose import jwt

        from backend.app.core.config import settings

        user = await _make_user(registry_session, username="legacy")
        legacy = jwt.encode(
            {
                "sub": user.id,
                "roles": ["user"],
                "type": "access",
                "exp": datetime.now(timezone.utc) + timedelta(minutes=60),
            },
            settings.secret_key,
            algorithm="HS256",
        )

        resp = await auth_client.get(
            "/api/athlete", headers={"Authorization": f"Bearer {legacy}"}
        )
        assert resp.status_code != 401

    async def test_pre_upgrade_token_dies_on_the_first_reset(self, auth_client, registry_session):
        """...and stops working as soon as the account's version moves."""
        from jose import jwt

        from backend.app.core.config import settings

        user = await _make_user(registry_session, username="legacy")
        legacy = jwt.encode(
            {
                "sub": user.id,
                "roles": ["user"],
                "type": "access",
                "exp": datetime.now(timezone.utc) + timedelta(minutes=60),
            },
            settings.secret_key,
            algorithm="HS256",
        )

        raw = await _issue_reset_token(registry_session, user)
        await auth_client.post(
            f"{_PREFIX}/reset-password", json={"token": raw, "new_password": _NEW_PW}
        )

        resp = await auth_client.get(
            "/api/athlete", headers={"Authorization": f"Bearer {legacy}"}
        )
        assert resp.status_code == 401

    @pytest.mark.parametrize("payload", [
        {"ver": "3"}, {"ver": None}, {"ver": 1.5}, {"ver": True},
    ])
    def test_non_integer_version_never_matches(self, payload):
        """Not something this server mints; fails closed rather than coercing."""
        assert claimed_token_version(payload) == -1
