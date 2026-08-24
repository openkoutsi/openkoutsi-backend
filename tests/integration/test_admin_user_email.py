"""Integration tests for the admin email escape hatch (issue #62).

Covers ``PATCH /api/admin/users/{user_id}/email``.

Users change their own address themselves, and that flow needs approval from the
address being left as well as the one being claimed — which is what stops
somebody holding only the password from relocating the account's password-reset
target. That same guarantee strands anyone whose old mailbox is gone, and before
this endpoint the only remedy was deleting the account and every activity in it.

Like ``test_change_email.py``, this builds its own client overriding only the
registry session: the shared ``client`` fixture waves every ``Bearer …`` value
through, so an admin-only route would look guarded when it isn't.
"""
import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from backend.app.core.auth import hash_password
from backend.app.core.encryption import set_user_encryption_context
from backend.app.db.registry import get_registry_session
from backend.app.db.user_session import get_user_session_factory, init_user_db
from backend.app.models.registry_orm import (
    EmailChangeToken,
    PersonalAccessToken,
    User,
)
from backend.app.models.user_orm import Athlete

_PREFIX = "/api/admin"
_AUTH = "/api/auth"
_GOOD_PW = "Testpass1234"


@pytest.fixture
async def auth_client(app, registry_session):
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


async def _make_user(
    registry_session, *, email: str | None, username: str | None = None,
    admin: bool = False,
) -> User:
    from backend.app.api.consent import CURRENT_CONSENT_VERSION

    now = datetime.now(timezone.utc)
    user = User(
        id=str(uuid.uuid4()),
        username=username,
        email=email,
        email_verified_at=now if email else None,
        password_hash=hash_password(_GOOD_PW),
        roles=json.dumps(["administrator"] if admin else ["user"]),
        consented_at=now,
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


async def _login(auth_client, identifier: str) -> str:
    resp = await auth_client.post(
        f"{_AUTH}/login", json={"username": identifier, "password": _GOOD_PW}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def _headers(access: str) -> dict:
    return {"Authorization": f"Bearer {access}"}


async def _reload(registry_session, user: User) -> User:
    result = await registry_session.execute(select(User).where(User.id == user.id))
    return result.scalar_one()


async def _admin_headers(auth_client, registry_session) -> dict:
    await _make_user(registry_session, email="admin@example.com", admin=True)
    return _headers(await _login(auth_client, "admin@example.com"))


class TestSetAndClear:
    async def test_sets_an_address_verified(
        self, auth_client, registry_session
    ):
        headers = await _admin_headers(auth_client, registry_session)
        victim = await _make_user(registry_session, email="stale@example.com")

        resp = await auth_client.patch(
            f"{_PREFIX}/users/{victim.id}/email",
            json={"email": "Rescued@Example.com"},
            headers=headers,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["email"] == "rescued@example.com"

        moved = await _reload(registry_session, victim)
        assert moved.email == "rescued@example.com"
        # Verified without a confirmation round: the point of the endpoint is
        # that the mailbox being replaced is unreachable.
        assert moved.email_verified_at is not None
        assert (await auth_client.post(
            f"{_AUTH}/login", json={"username": "rescued@example.com", "password": _GOOD_PW}
        )).status_code == 200

    async def test_clearing_removes_the_address_and_its_verification(
        self, auth_client, registry_session
    ):
        headers = await _admin_headers(auth_client, registry_session)
        victim = await _make_user(
            registry_session, email="hijacked@example.com", username="victim"
        )

        resp = await auth_client.patch(
            f"{_PREFIX}/users/{victim.id}/email", json={"email": None}, headers=headers
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["email"] is None

        cleared = await _reload(registry_session, victim)
        assert cleared.email is None
        assert cleared.email_verified_at is None
        # The stolen address no longer reaches the account at all.
        assert (await auth_client.post(
            f"{_AUTH}/login",
            json={"username": "hijacked@example.com", "password": _GOOD_PW},
        )).status_code == 401
        # The username still signs in, so the account is recoverable.
        assert (await auth_client.post(
            f"{_AUTH}/login", json={"username": "victim", "password": _GOOD_PW}
        )).status_code == 200

    async def test_address_held_by_another_account_is_409(
        self, auth_client, registry_session
    ):
        headers = await _admin_headers(auth_client, registry_session)
        await _make_user(registry_session, email="taken@example.com")
        victim = await _make_user(registry_session, email="stale@example.com")

        resp = await auth_client.patch(
            f"{_PREFIX}/users/{victim.id}/email",
            json={"email": "taken@example.com"},
            headers=headers,
        )
        assert resp.status_code == 409
        assert (await _reload(registry_session, victim)).email == "stale@example.com"

    async def test_unknown_user_is_404(self, auth_client, registry_session):
        headers = await _admin_headers(auth_client, registry_session)
        resp = await auth_client.patch(
            f"{_PREFIX}/users/{uuid.uuid4()}/email",
            json={"email": "x@example.com"},
            headers=headers,
        )
        assert resp.status_code == 404


class TestItIsARecoveryAction:
    async def test_it_signs_the_current_holder_out(
        self, auth_client, registry_session
    ):
        """Assume the account is in the wrong hands — that is why this exists."""
        headers = await _admin_headers(auth_client, registry_session)
        victim = await _make_user(registry_session, email="hijacked@example.com")
        holder = _headers(await _login(auth_client, "hijacked@example.com"))
        assert (await auth_client.get(f"{_AUTH}/account", headers=holder)).status_code == 200

        await auth_client.patch(
            f"{_PREFIX}/users/{victim.id}/email", json={"email": None}, headers=headers
        )

        assert (await auth_client.get(
            f"{_AUTH}/account", headers=holder
        )).status_code == 401

    async def test_it_revokes_personal_access_tokens(
        self, auth_client, registry_session
    ):
        headers = await _admin_headers(auth_client, registry_session)
        victim = await _make_user(registry_session, email="hijacked@example.com")
        registry_session.add(PersonalAccessToken(
            id=str(uuid.uuid4()),
            user_id=victim.id,
            token_hash=hashlib.sha256(b"pat").hexdigest(),
            name="attacker's sync",
            scopes=json.dumps(["activities:read"]),
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        ))
        await registry_session.commit()

        await auth_client.patch(
            f"{_PREFIX}/users/{victim.id}/email", json={"email": None}, headers=headers
        )

        token = (await registry_session.execute(
            select(PersonalAccessToken).where(PersonalAccessToken.user_id == victim.id)
        )).scalar_one()
        assert token.revoked_at is not None

    async def test_it_voids_a_change_already_in_flight(
        self, auth_client, registry_session
    ):
        """That change was authorised against the address being replaced."""
        headers = await _admin_headers(auth_client, registry_session)
        victim = await _make_user(registry_session, email="hijacked@example.com")
        raw = "pending-token"
        registry_session.add(EmailChangeToken(
            id=str(uuid.uuid4()),
            user_id=victim.id,
            new_token_hash=hashlib.sha256(raw.encode()).hexdigest(),
            new_email="attacker@example.com",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=12),
        ))
        await registry_session.commit()

        await auth_client.patch(
            f"{_PREFIX}/users/{victim.id}/email",
            json={"email": "rescued@example.com"},
            headers=headers,
        )

        row = (await registry_session.execute(
            select(EmailChangeToken).where(EmailChangeToken.user_id == victim.id)
        )).scalar_one()
        assert row.used_at is not None
        # And the link that already reached the attacker is dead.
        assert (await auth_client.post(
            f"{_AUTH}/confirm-email-change", json={"token": raw}
        )).status_code == 400
        assert (await _reload(registry_session, victim)).email == "rescued@example.com"


class TestItIsAdminOnly:
    async def test_a_normal_user_is_refused(self, auth_client, registry_session):
        victim = await _make_user(registry_session, email="victim@example.com")
        await _make_user(registry_session, email="nosy@example.com")
        headers = _headers(await _login(auth_client, "nosy@example.com"))

        resp = await auth_client.patch(
            f"{_PREFIX}/users/{victim.id}/email",
            json={"email": "mine@example.com"},
            headers=headers,
        )
        assert resp.status_code == 403
        assert (await _reload(registry_session, victim)).email == "victim@example.com"

    async def test_unauthenticated_is_401(self, auth_client, registry_session):
        victim = await _make_user(registry_session, email="victim@example.com")
        resp = await auth_client.patch(
            f"{_PREFIX}/users/{victim.id}/email", json={"email": "mine@example.com"}
        )
        assert resp.status_code == 401
