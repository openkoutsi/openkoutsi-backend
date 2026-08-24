"""Integration tests for changing (or setting) the account's email address (issue #62).

Covers ``/auth/account``, ``/auth/change-email``, ``/auth/confirm-email-change``
and ``/auth/cancel-email-change``.

Like ``test_session_invalidation.py``, this module builds its own client that
overrides **only** the registry session, so the real identity path runs. The
shared ``client`` fixture waves every ``Bearer …`` value through, which would
make the password check here look like it works when it doesn't.

The email provider is replaced with a recording fake through the
``get_email_provider_dep`` override, the same seam ``test_signup_reset.py`` uses.
"""
import hashlib
import json
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from backend.app.api.auth import get_email_provider_dep
from backend.app.core.auth import hash_password
from backend.app.core.encryption import set_user_encryption_context
from backend.app.db.registry import get_registry_session
from backend.app.db.user_session import get_user_session_factory, init_user_db
from backend.app.models.registry_orm import (
    EmailChangeToken,
    EmailVerificationToken,
    User,
)
from backend.app.models.user_orm import Athlete

_PREFIX = "/api/auth"
_GOOD_PW = "Testpass1234"
_OLD_EMAIL = "old@example.com"
_NEW_EMAIL = "new@example.com"
_TOKEN_RE = re.compile(r"token=([A-Za-z0-9_\-]+)")


# ── Fixtures and helpers ────────────────────────────────────────────────────


class _FakeProvider:
    def __init__(self, configured: bool = True):
        self._configured = configured
        self.sent: list = []

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def send(self, message) -> str:
        self.sent.append(message)
        return "fake-message-id"

    def to(self, address: str) -> list:
        """Messages sent to an address, matched case-insensitively as mail is."""
        return [m for m in self.sent if m.to.lower() == address.lower()]


@pytest.fixture
async def auth_client(app, registry_session):
    """A client on the **real** identity path — only the registry is overridden."""
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


def _use_provider(app, configured: bool = True) -> _FakeProvider:
    fake = _FakeProvider(configured=configured)
    app.dependency_overrides[get_email_provider_dep] = lambda: fake
    return fake


async def _make_user(
    registry_session,
    *,
    username: str | None = None,
    email: str | None = _OLD_EMAIL,
    verified: bool = True,
) -> User:
    """A real account: registry row, own database, athlete profile.

    ``email=None`` builds the invite-created shape — a username and no address
    at all, which is what a first-time *set* starts from.
    """
    from backend.app.api.consent import CURRENT_CONSENT_VERSION

    now = datetime.now(timezone.utc)
    user = User(
        id=str(uuid.uuid4()),
        username=username,
        email=email,
        email_verified_at=now if (email and verified) else None,
        password_hash=hash_password(_GOOD_PW),
        roles=json.dumps(["user"]),
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


async def _login(auth_client, identifier: str, password: str = _GOOD_PW) -> str:
    resp = await auth_client.post(
        f"{_PREFIX}/login", json={"username": identifier, "password": password}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def _headers(access: str) -> dict:
    return {"Authorization": f"Bearer {access}"}


def _extract_token(message) -> str:
    m = _TOKEN_RE.search(message.text) or _TOKEN_RE.search(message.html)
    assert m, "no token in sent message"
    return m.group(1)


async def _reload(registry_session, user: User) -> User:
    """Re-read a user row from the registry.

    Not ``session.refresh``: any endpoint behind ``get_current_user`` closes the
    registry session on its way in (core/auth.py — it releases the pool slot
    before the per-user session is opened). Production hands each request its own
    session, but the fixture shares one with the test, so that close expunges
    whatever the test is holding. Re-querying gets the committed row either way.
    """
    result = await registry_session.execute(select(User).where(User.id == user.id))
    return result.scalar_one()


class _Change(NamedTuple):
    provider: _FakeProvider
    new_token: str
    # None only for a first-time set, where there is no address to approve from.
    old_token: str | None


async def _request_change(
    auth_client, app, user_email: str, new_email: str = _NEW_EMAIL
) -> _Change:
    """Log in, ask for the change, and collect the token mailed to each side."""
    fake = _use_provider(app)
    access = await _login(auth_client, user_email)
    resp = await auth_client.post(
        f"{_PREFIX}/change-email",
        json={"new_email": new_email, "password": _GOOD_PW},
        headers=_headers(access),
    )
    assert resp.status_code == 202, resp.text
    to_old = fake.to(user_email)
    return _Change(
        provider=fake,
        new_token=_extract_token(fake.to(new_email)[0]),
        old_token=_extract_token(to_old[0]) if to_old else None,
    )


async def _confirm(auth_client, token: str):
    return await auth_client.post(
        f"{_PREFIX}/confirm-email-change", json={"token": token}
    )


async def _complete(auth_client, change: _Change):
    """Open both links. Returns the response that actually finished the change."""
    resp = await _confirm(auth_client, change.new_token)
    if change.old_token is None:
        return resp
    assert resp.status_code == 200, resp.text
    assert resp.json()["complete"] is False
    return await _confirm(auth_client, change.old_token)


# ── The happy path ──────────────────────────────────────────────────────────


class TestChangeEmail:
    async def test_confirm_moves_the_account(self, auth_client, app, registry_session):
        user = await _make_user(registry_session)
        change = await _request_change(auth_client, app, _OLD_EMAIL)

        # Nothing moves until both links are opened.
        assert (await _reload(registry_session, user)).email == _OLD_EMAIL

        half = await _confirm(auth_client, change.new_token)
        assert half.status_code == 200, half.text
        assert half.json() == {
            "complete": False, "awaiting": "old", "new_email": _NEW_EMAIL
        }
        assert (await _reload(registry_session, user)).email == _OLD_EMAIL

        resp = await _confirm(auth_client, change.old_token)
        assert resp.status_code == 200, resp.text
        assert resp.json()["complete"] is True

        moved = await _reload(registry_session, user)
        assert moved.email == _NEW_EMAIL
        assert moved.email_verified_at is not None

        token_row = (await registry_session.execute(
            select(EmailChangeToken).where(EmailChangeToken.user_id == user.id)
        )).scalar_one()
        assert token_row.used_at is not None

    async def test_new_address_signs_in_and_old_one_stops(
        self, auth_client, app, registry_session
    ):
        await _make_user(registry_session)
        await _complete(
            auth_client, await _request_change(auth_client, app, _OLD_EMAIL)
        )

        assert (await auth_client.post(
            f"{_PREFIX}/login", json={"username": _NEW_EMAIL, "password": _GOOD_PW}
        )).status_code == 200
        assert (await auth_client.post(
            f"{_PREFIX}/login", json={"username": _OLD_EMAIL, "password": _GOOD_PW}
        )).status_code == 401

    async def test_address_is_lowercased(self, auth_client, app, registry_session):
        user = await _make_user(registry_session)
        change = await _request_change(
            auth_client, app, _OLD_EMAIL, new_email="MiXeD@Example.COM"
        )
        # Normalised on the way out, not only on the way into the row.
        assert change.provider.to("mixed@example.com")[0].to == "mixed@example.com"

        await _complete(auth_client, change)
        assert (await _reload(registry_session, user)).email == "mixed@example.com"

    async def test_confirm_needs_no_session(self, auth_client, app, registry_session):
        """The link is opened in the new mailbox, routinely on another device."""
        await _make_user(registry_session)
        change = await _request_change(auth_client, app, _OLD_EMAIL)
        assert (await _confirm(auth_client, change.new_token)).status_code == 200
        assert (await _confirm(auth_client, change.old_token)).status_code == 200

    async def test_sessions_survive_the_change(self, auth_client, app, registry_session):
        """Decided in #62: this is not a credential change, so it signs nobody out."""
        await _make_user(registry_session)
        fake = _use_provider(app)
        access = await _login(auth_client, _OLD_EMAIL)
        resp = await auth_client.post(
            f"{_PREFIX}/change-email",
            json={"new_email": _NEW_EMAIL, "password": _GOOD_PW},
            headers=_headers(access),
        )
        assert resp.status_code == 202
        await _confirm(auth_client, _extract_token(fake.to(_NEW_EMAIL)[0]))
        await _confirm(auth_client, _extract_token(fake.to(_OLD_EMAIL)[0]))

        still_good = await auth_client.get(f"{_PREFIX}/account", headers=_headers(access))
        assert still_good.status_code == 200
        assert still_good.json()["email"] == _NEW_EMAIL


# ── Who is told what ────────────────────────────────────────────────────────


class TestNotifications:
    async def test_old_address_gets_its_own_link(
        self, auth_client, app, registry_session
    ):
        await _make_user(registry_session)
        change = await _request_change(auth_client, app, _OLD_EMAIL)

        assert len(change.provider.to(_NEW_EMAIL)) == 1
        authorisations = change.provider.to(_OLD_EMAIL)
        assert len(authorisations) == 1
        # It carries a link, and names where the account is being moved to.
        assert "token=" in authorisations[0].text
        assert _NEW_EMAIL in authorisations[0].text
        # Two different secrets. One value satisfying both sides would let
        # whoever reads either mailbox finish alone.
        assert change.old_token != change.new_token

    async def test_first_time_set_notifies_only_the_new_address(
        self, auth_client, app, registry_session
    ):
        """An invite account has no old mailbox to warn."""
        await _make_user(registry_session, username="invited", email=None)
        fake = _use_provider(app)
        access = await _login(auth_client, "invited")
        resp = await auth_client.post(
            f"{_PREFIX}/change-email",
            json={"new_email": _NEW_EMAIL, "password": _GOOD_PW},
            headers=_headers(access),
        )
        assert resp.status_code == 202
        assert len(fake.sent) == 1
        assert fake.sent[0].to == _NEW_EMAIL


# ── Setting an address on an invite-created account ─────────────────────────


class TestFirstTimeSet:
    async def test_invited_account_gains_an_address(
        self, auth_client, app, registry_session
    ):
        user = await _make_user(registry_session, username="invited", email=None)
        fake = _use_provider(app)
        access = await _login(auth_client, "invited")
        await auth_client.post(
            f"{_PREFIX}/change-email",
            json={"new_email": _NEW_EMAIL, "password": _GOOD_PW},
            headers=_headers(access),
        )
        raw = _extract_token(fake.to(_NEW_EMAIL)[0])
        done = await _confirm(auth_client, raw)
        assert done.status_code == 200, done.text
        # No address to approve from, so the new side alone finishes it.
        assert done.json()["complete"] is True

        moved = await _reload(registry_session, user)
        assert moved.email == _NEW_EMAIL
        assert moved.email_verified_at is not None
        # The username still works — the account gained an identifier, it did
        # not trade one for the other.
        assert (await auth_client.post(
            f"{_PREFIX}/login", json={"username": "invited", "password": _GOOD_PW}
        )).status_code == 200

    async def test_password_reset_now_reaches_them(
        self, auth_client, app, registry_session
    ):
        """The point of letting an invite account set an address at all."""
        await _make_user(registry_session, username="invited", email=None)
        fake = _use_provider(app)
        access = await _login(auth_client, "invited")
        await auth_client.post(
            f"{_PREFIX}/change-email",
            json={"new_email": _NEW_EMAIL, "password": _GOOD_PW},
            headers=_headers(access),
        )
        await _confirm(auth_client, _extract_token(fake.to(_NEW_EMAIL)[0]))

        fake.sent.clear()
        resp = await auth_client.post(
            f"{_PREFIX}/request-password-reset", json={"email": _NEW_EMAIL}
        )
        assert resp.status_code == 200
        assert len(fake.to(_NEW_EMAIL)) == 1


# ── Refusals ────────────────────────────────────────────────────────────────


class TestRefusals:
    async def test_wrong_password_is_401_and_sends_nothing(
        self, auth_client, app, registry_session
    ):
        await _make_user(registry_session)
        fake = _use_provider(app)
        access = await _login(auth_client, _OLD_EMAIL)
        resp = await auth_client.post(
            f"{_PREFIX}/change-email",
            json={"new_email": _NEW_EMAIL, "password": "Wrongpass1234"},
            headers=_headers(access),
        )
        assert resp.status_code == 401
        assert fake.sent == []

    async def test_unauthenticated_is_401(self, auth_client, app, registry_session):
        _use_provider(app)
        resp = await auth_client.post(
            f"{_PREFIX}/change-email",
            json={"new_email": _NEW_EMAIL, "password": _GOOD_PW},
        )
        assert resp.status_code == 401

    async def test_unconfigured_provider_is_404(
        self, auth_client, app, registry_session
    ):
        await _make_user(registry_session)
        fake = _use_provider(app, configured=False)
        access = await _login(auth_client, _OLD_EMAIL)
        resp = await auth_client.post(
            f"{_PREFIX}/change-email",
            json={"new_email": _NEW_EMAIL, "password": _GOOD_PW},
            headers=_headers(access),
        )
        assert resp.status_code == 404
        assert fake.sent == []

    async def test_malformed_address_is_422(self, auth_client, app, registry_session):
        await _make_user(registry_session)
        _use_provider(app)
        access = await _login(auth_client, _OLD_EMAIL)
        resp = await auth_client.post(
            f"{_PREFIX}/change-email",
            json={"new_email": "not-an-address", "password": _GOOD_PW},
            headers=_headers(access),
        )
        assert resp.status_code == 422


# ── No account enumeration ──────────────────────────────────────────────────


class TestNoEnumeration:
    async def test_taken_address_acks_and_sends_nothing(
        self, auth_client, app, registry_session
    ):
        await _make_user(registry_session)
        await _make_user(registry_session, email="taken@example.com")
        fake = _use_provider(app)
        access = await _login(auth_client, _OLD_EMAIL)

        resp = await auth_client.post(
            f"{_PREFIX}/change-email",
            json={"new_email": "taken@example.com", "password": _GOOD_PW},
            headers=_headers(access),
        )
        assert resp.status_code == 202
        assert fake.sent == []

    async def test_taken_and_free_read_identically(
        self, auth_client, app, registry_session
    ):
        """The whole point: the response cannot be used to probe for accounts."""
        await _make_user(registry_session)
        await _make_user(registry_session, email="taken@example.com")
        _use_provider(app)
        access = await _login(auth_client, _OLD_EMAIL)

        taken = await auth_client.post(
            f"{_PREFIX}/change-email",
            json={"new_email": "taken@example.com", "password": _GOOD_PW},
            headers=_headers(access),
        )
        free = await auth_client.post(
            f"{_PREFIX}/change-email",
            json={"new_email": "free@example.com", "password": _GOOD_PW},
            headers=_headers(access),
        )
        assert taken.status_code == free.status_code == 202
        assert taken.json() == free.json()

    async def test_own_current_address_is_a_silent_no_op(
        self, auth_client, app, registry_session
    ):
        await _make_user(registry_session)
        fake = _use_provider(app)
        access = await _login(auth_client, _OLD_EMAIL)
        resp = await auth_client.post(
            f"{_PREFIX}/change-email",
            json={"new_email": _OLD_EMAIL.upper(), "password": _GOOD_PW},
            headers=_headers(access),
        )
        assert resp.status_code == 202
        assert fake.sent == []


# ── Token handling ──────────────────────────────────────────────────────────


class TestTokens:
    async def test_token_is_single_use(self, auth_client, app, registry_session):
        await _make_user(registry_session)
        change = await _request_change(auth_client, app, _OLD_EMAIL)
        assert (await _complete(auth_client, change)).status_code == 200
        # Both are spent once the change lands.
        assert (await _confirm(auth_client, change.new_token)).status_code == 400
        assert (await _confirm(auth_client, change.old_token)).status_code == 400

    async def test_unknown_token_is_400(self, auth_client, registry_session):
        resp = await auth_client.post(
            f"{_PREFIX}/confirm-email-change", json={"token": secrets.token_urlsafe(32)}
        )
        assert resp.status_code == 400

    async def test_expired_token_is_400(self, auth_client, app, registry_session):
        user = await _make_user(registry_session)
        raw = secrets.token_urlsafe(32)
        registry_session.add(EmailChangeToken(
            id=str(uuid.uuid4()),
            user_id=user.id,
            new_token_hash=hashlib.sha256(raw.encode()).hexdigest(),
            new_email=_NEW_EMAIL,
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        ))
        await registry_session.commit()

        resp = await auth_client.post(
            f"{_PREFIX}/confirm-email-change", json={"token": raw}
        )
        assert resp.status_code == 400
        assert (await _reload(registry_session, user)).email == _OLD_EMAIL

    async def test_second_request_supersedes_the_first(
        self, auth_client, app, registry_session
    ):
        await _make_user(registry_session)
        fake = _use_provider(app)
        access = await _login(auth_client, _OLD_EMAIL)
        for target in ("first@example.com", "second@example.com"):
            await auth_client.post(
                f"{_PREFIX}/change-email",
                json={"new_email": target, "password": _GOOD_PW},
                headers=_headers(access),
            )
        first = _extract_token(fake.to("first@example.com")[0])
        second = _extract_token(fake.to("second@example.com")[0])
        # Two authorisations reached the old address; the live one is the newest.
        old_second = _extract_token(fake.to(_OLD_EMAIL)[1])

        assert (await _confirm(auth_client, first)).status_code == 400
        assert (await _confirm(auth_client, second)).status_code == 200
        assert (await _confirm(auth_client, old_second)).status_code == 200

    async def test_address_claimed_before_confirmation_is_409(
        self, auth_client, app, registry_session
    ):
        user = await _make_user(registry_session)
        change = await _request_change(auth_client, app, _OLD_EMAIL)
        assert (await _confirm(auth_client, change.new_token)).status_code == 200
        # Somebody else takes the address while the second link is still live.
        await _make_user(registry_session, email=_NEW_EMAIL)

        resp = await _confirm(auth_client, change.old_token)
        assert resp.status_code == 409
        assert (await _reload(registry_session, user)).email == _OLD_EMAIL


# ── GET /account and cancelling ─────────────────────────────────────────────


class TestAccountAndCancel:
    async def test_account_reports_identifiers(
        self, auth_client, app, registry_session
    ):
        await _make_user(registry_session, username="both")
        access = await _login(auth_client, _OLD_EMAIL)
        body = (await auth_client.get(
            f"{_PREFIX}/account", headers=_headers(access)
        )).json()
        assert body == {
            "username": "both",
            "email": _OLD_EMAIL,
            "email_verified": True,
            "pending_email": None,
            "pending_requires_old": False,
            "pending_confirmed_new": False,
            "pending_confirmed_old": False,
        }

    async def test_account_reports_a_pending_change(
        self, auth_client, app, registry_session
    ):
        await _make_user(registry_session)
        fake = _use_provider(app)
        access = await _login(auth_client, _OLD_EMAIL)
        await auth_client.post(
            f"{_PREFIX}/change-email",
            json={"new_email": _NEW_EMAIL, "password": _GOOD_PW},
            headers=_headers(access),
        )
        body = (await auth_client.get(
            f"{_PREFIX}/account", headers=_headers(access)
        )).json()
        assert body["email"] == _OLD_EMAIL
        assert body["pending_email"] == _NEW_EMAIL
        # Both mailboxes still outstanding, and the card needs to know the old
        # one counts at all.
        assert body["pending_requires_old"] is True
        assert body["pending_confirmed_new"] is False
        assert body["pending_confirmed_old"] is False

    async def test_expired_pending_change_is_not_reported(
        self, auth_client, app, registry_session
    ):
        user = await _make_user(registry_session)
        registry_session.add(EmailChangeToken(
            id=str(uuid.uuid4()),
            user_id=user.id,
            new_token_hash=hashlib.sha256(b"stale").hexdigest(),
            new_email=_NEW_EMAIL,
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        ))
        await registry_session.commit()
        access = await _login(auth_client, _OLD_EMAIL)
        body = (await auth_client.get(
            f"{_PREFIX}/account", headers=_headers(access)
        )).json()
        assert body["pending_email"] is None

    async def test_cancel_clears_the_pending_change(
        self, auth_client, app, registry_session
    ):
        await _make_user(registry_session)
        change = await _request_change(auth_client, app, _OLD_EMAIL)
        access = await _login(auth_client, _OLD_EMAIL)

        assert (await auth_client.post(
            f"{_PREFIX}/cancel-email-change", headers=_headers(access)
        )).status_code == 204
        body = (await auth_client.get(
            f"{_PREFIX}/account", headers=_headers(access)
        )).json()
        assert body["pending_email"] is None
        # Both links that already reached an inbox are dead.
        assert (await _confirm(auth_client, change.new_token)).status_code == 400
        assert (await _confirm(auth_client, change.old_token)).status_code == 400

    async def test_cancel_with_nothing_pending_succeeds(
        self, auth_client, app, registry_session
    ):
        await _make_user(registry_session)
        access = await _login(auth_client, _OLD_EMAIL)
        assert (await auth_client.post(
            f"{_PREFIX}/cancel-email-change", headers=_headers(access)
        )).status_code == 204


# ── Dual confirmation: the property the whole design rests on ───────────────


class TestDualConfirmation:
    """Why both sides are required.

    Passwords are set through reset tokens mailed to ``users.email``, and there
    is no authenticated change-password endpoint, so the address *is* the
    account's recovery channel. A one-sided change would let anyone holding the
    password move that channel and then take the account permanently. These
    tests pin the behaviour that stops it.
    """

    async def test_new_side_alone_does_not_move_the_account(
        self, auth_client, app, registry_session
    ):
        user = await _make_user(registry_session)
        change = await _request_change(auth_client, app, _OLD_EMAIL)

        resp = await _confirm(auth_client, change.new_token)
        assert resp.status_code == 200
        assert resp.json()["complete"] is False

        unmoved = await _reload(registry_session, user)
        assert unmoved.email == _OLD_EMAIL
        # And the old address still signs in.
        assert (await auth_client.post(
            f"{_PREFIX}/login", json={"username": _OLD_EMAIL, "password": _GOOD_PW}
        )).status_code == 200

    async def test_old_side_alone_does_not_move_the_account(
        self, auth_client, app, registry_session
    ):
        user = await _make_user(registry_session)
        change = await _request_change(auth_client, app, _OLD_EMAIL)

        resp = await _confirm(auth_client, change.old_token)
        assert resp.status_code == 200
        assert resp.json() == {
            "complete": False, "awaiting": "new", "new_email": _NEW_EMAIL
        }
        assert (await _reload(registry_session, user)).email == _OLD_EMAIL

    async def test_same_token_twice_completes_nothing(
        self, auth_client, app, registry_session
    ):
        """The attack this design exists to stop.

        Someone who reads only the new mailbox holds one secret. Replaying it
        must not stand in for the approval they cannot reach — otherwise the
        second confirmation is a notification with a button on it.
        """
        user = await _make_user(registry_session)
        change = await _request_change(auth_client, app, _OLD_EMAIL)

        for _ in range(3):
            resp = await _confirm(auth_client, change.new_token)
            assert resp.status_code == 200
            assert resp.json()["complete"] is False
            assert resp.json()["awaiting"] == "old"

        assert (await _reload(registry_session, user)).email == _OLD_EMAIL

    async def test_either_order_works(self, auth_client, app, registry_session):
        user = await _make_user(registry_session)
        change = await _request_change(auth_client, app, _OLD_EMAIL)

        assert (await _confirm(auth_client, change.old_token)).status_code == 200
        final = await _confirm(auth_client, change.new_token)
        assert final.status_code == 200
        assert final.json()["complete"] is True
        assert (await _reload(registry_session, user)).email == _NEW_EMAIL

    async def test_account_reports_each_side_as_it_lands(
        self, auth_client, app, registry_session
    ):
        await _make_user(registry_session)
        change = await _request_change(auth_client, app, _OLD_EMAIL)
        access = await _login(auth_client, _OLD_EMAIL)

        await _confirm(auth_client, change.new_token)
        body = (await auth_client.get(
            f"{_PREFIX}/account", headers=_headers(access)
        )).json()
        assert body["pending_confirmed_new"] is True
        assert body["pending_confirmed_old"] is False

    async def test_first_time_set_needs_no_old_side(
        self, auth_client, app, registry_session
    ):
        await _make_user(registry_session, username="invited", email=None)
        fake = _use_provider(app)
        access = await _login(auth_client, "invited")
        await auth_client.post(
            f"{_PREFIX}/change-email",
            json={"new_email": _NEW_EMAIL, "password": _GOOD_PW},
            headers=_headers(access),
        )
        body = (await auth_client.get(
            f"{_PREFIX}/account", headers=_headers(access)
        )).json()
        assert body["pending_requires_old"] is False

    async def test_expiry_covers_the_whole_change(
        self, auth_client, app, registry_session
    ):
        """One side confirmed does not keep the other alive past the deadline."""
        user = await _make_user(registry_session)
        change = await _request_change(auth_client, app, _OLD_EMAIL)
        assert (await _confirm(auth_client, change.new_token)).status_code == 200

        row = (await registry_session.execute(
            select(EmailChangeToken).where(EmailChangeToken.user_id == user.id)
        )).scalar_one()
        row.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        await registry_session.commit()

        assert (await _confirm(auth_client, change.old_token)).status_code == 400
        assert (await _reload(registry_session, user)).email == _OLD_EMAIL


# ── Recovery has to actually recover ────────────────────────────────────────


class TestPasswordResetDisarmsAChange:
    """A reset withdraws in-flight changes as well as credentials (issue #62).

    Found in review. The dual confirmation closes the front door, but leaving a
    pending change armed reopened it on the way out: an attacker holding the
    password arms a move and confirms the side they own, and the victim's own
    recovery — the remedy the authorisation email recommends by name — used to
    leave the other approval live in their inbox for the rest of the day.
    """

    async def _reset_password(self, auth_client, app, email: str) -> None:
        fake = _use_provider(app)
        resp = await auth_client.post(
            f"{_PREFIX}/request-password-reset", json={"email": email}
        )
        assert resp.status_code == 200, resp.text
        raw = _extract_token(fake.to(email)[0])
        done = await auth_client.post(
            f"{_PREFIX}/reset-password",
            json={"token": raw, "new_password": "Brandnew12345"},
        )
        assert done.status_code == 204, done.text

    async def test_the_old_side_link_is_dead_after_a_reset(
        self, auth_client, app, registry_session
    ):
        user = await _make_user(registry_session)
        change = await _request_change(auth_client, app, _OLD_EMAIL)
        # The attacker holds the new mailbox and confirms their half at once.
        assert (await _confirm(auth_client, change.new_token)).status_code == 200

        # The victim does what the authorisation email told them to.
        await self._reset_password(auth_client, app, _OLD_EMAIL)

        # The approval sitting in their inbox no longer completes anything.
        assert (await _confirm(auth_client, change.old_token)).status_code == 400
        assert (await _reload(registry_session, user)).email == _OLD_EMAIL

    async def test_the_new_side_link_is_dead_too(
        self, auth_client, app, registry_session
    ):
        """Neither half survives, whichever the attacker had already opened."""
        user = await _make_user(registry_session)
        change = await _request_change(auth_client, app, _OLD_EMAIL)
        await self._reset_password(auth_client, app, _OLD_EMAIL)

        assert (await _confirm(auth_client, change.new_token)).status_code == 400
        assert (await _reload(registry_session, user)).email == _OLD_EMAIL

    async def test_account_stops_reporting_the_change(
        self, auth_client, app, registry_session
    ):
        await _make_user(registry_session)
        await _request_change(auth_client, app, _OLD_EMAIL)
        await self._reset_password(auth_client, app, _OLD_EMAIL)

        access = await _login(auth_client, _OLD_EMAIL, "Brandnew12345")
        body = (await auth_client.get(
            f"{_PREFIX}/account", headers=_headers(access)
        )).json()
        assert body["pending_email"] is None

    async def test_a_reset_with_nothing_pending_still_works(
        self, auth_client, app, registry_session
    ):
        await _make_user(registry_session)
        await self._reset_password(auth_client, app, _OLD_EMAIL)
        assert (await auth_client.post(
            f"{_PREFIX}/login",
            json={"username": _OLD_EMAIL, "password": "Brandnew12345"},
        )).status_code == 200


# ── An abandoned signup must not squat on an address ────────────────────────


class TestAbandonedSignupStubs:
    """``signup`` reuses a stub that never verified, so a change may claim one too.

    Refusing would make this flow stricter than the one that created the
    obstruction — permanently, since nothing expires such a row, and invisibly,
    since the uniform acknowledgement gives the user no reason why no link ever
    arrives. With self-serve signup on it would also let anyone deny an address
    to its real owner for good by signing up and walking away.
    """

    async def _stub(self, registry_session, email: str, *, token_hours: float | None):
        """An unverified signup row, optionally with a verification token."""
        stub = User(
            id=str(uuid.uuid4()),
            email=email,
            password_hash=hash_password(_GOOD_PW),
            roles=json.dumps(["user"]),
        )
        registry_session.add(stub)
        await registry_session.flush()
        if token_hours is not None:
            registry_session.add(EmailVerificationToken(
                id=str(uuid.uuid4()),
                user_id=stub.id,
                token_hash=hashlib.sha256(f"v-{email}".encode()).hexdigest(),
                expires_at=datetime.now(timezone.utc) + timedelta(hours=token_hours),
            ))
        await registry_session.commit()
        return stub

    async def test_a_dead_stub_does_not_block_and_is_reaped(
        self, auth_client, app, registry_session
    ):
        user = await _make_user(registry_session)
        stub = await self._stub(registry_session, _NEW_EMAIL, token_hours=-1)

        change = await _request_change(auth_client, app, _OLD_EMAIL)
        # It was not silently swallowed: a link actually went out.
        assert len(change.provider.to(_NEW_EMAIL)) == 1
        assert (await _complete(auth_client, change)).status_code == 200

        assert (await _reload(registry_session, user)).email == _NEW_EMAIL
        gone = (await registry_session.execute(
            select(User).where(User.id == stub.id)
        )).scalar_one_or_none()
        assert gone is None

    async def test_a_stub_with_no_token_at_all_does_not_block(
        self, auth_client, app, registry_session
    ):
        user = await _make_user(registry_session)
        await self._stub(registry_session, _NEW_EMAIL, token_hours=None)
        change = await _request_change(auth_client, app, _OLD_EMAIL)
        assert (await _complete(auth_client, change)).status_code == 200
        assert (await _reload(registry_session, user)).email == _NEW_EMAIL

    async def test_a_signup_still_in_progress_does_block(
        self, auth_client, app, registry_session
    ):
        """A live token means somebody is mid-signup, not that they walked away."""
        await _make_user(registry_session)
        await self._stub(registry_session, _NEW_EMAIL, token_hours=1)
        fake = _use_provider(app)
        access = await _login(auth_client, _OLD_EMAIL)
        resp = await auth_client.post(
            f"{_PREFIX}/change-email",
            json={"new_email": _NEW_EMAIL, "password": _GOOD_PW},
            headers=_headers(access),
        )
        assert resp.status_code == 202
        assert fake.sent == []

    async def test_a_verified_account_still_blocks(
        self, auth_client, app, registry_session
    ):
        await _make_user(registry_session)
        await _make_user(registry_session, email=_NEW_EMAIL)
        fake = _use_provider(app)
        access = await _login(auth_client, _OLD_EMAIL)
        resp = await auth_client.post(
            f"{_PREFIX}/change-email",
            json={"new_email": _NEW_EMAIL, "password": _GOOD_PW},
            headers=_headers(access),
        )
        assert resp.status_code == 202
        assert fake.sent == []
