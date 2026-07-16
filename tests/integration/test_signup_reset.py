"""
Integration tests for self-serve email signup + password reset (issue #15).

Covers the ``/auth/signup``, ``/auth/verify-email`` and
``/auth/request-password-reset`` endpoints, login by verified email, the
``allow_self_signup`` toggle, and graceful degradation when no email provider is
configured. The email provider is replaced with a recording fake via the
``get_email_provider_dep`` dependency override.
"""
import hashlib
import json
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from backend.app.api.auth import _create_user_profile, get_email_provider_dep
from backend.app.core.auth import hash_password
from backend.app.db.user_session import get_user_session_factory
from backend.app.models.registry_orm import (
    EmailVerificationToken,
    InstanceSettings,
    PasswordResetToken,
    User,
)
from backend.app.models.user_orm import Athlete

_PREFIX = "/api/auth"
_GOOD_PW = "Testpass1234"
_TOKEN_RE = re.compile(r"token=([A-Za-z0-9_\-]+)")


# ── Fake email provider ─────────────────────────────────────────────────────


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


def _use_provider(app, configured: bool = True) -> _FakeProvider:
    fake = _FakeProvider(configured=configured)
    app.dependency_overrides[get_email_provider_dep] = lambda: fake
    return fake


async def _enable_signup(registry_session) -> None:
    registry_session.add(InstanceSettings(id=1, allow_self_signup=True))
    await registry_session.commit()


async def _add_verified_user(registry_session, email: str) -> User:
    user = User(
        id=str(uuid.uuid4()),
        email=email,
        password_hash=hash_password(_GOOD_PW),
        roles=json.dumps(["user"]),
        email_verified_at=datetime.now(timezone.utc),
    )
    registry_session.add(user)
    await registry_session.commit()
    return user


def _extract_token(message) -> str:
    m = _TOKEN_RE.search(message.text) or _TOKEN_RE.search(message.html)
    assert m, "no token in sent message"
    return m.group(1)


# ── /signup ─────────────────────────────────────────────────────────────────


class TestSignup:
    async def test_disabled_toggle_returns_404(self, client, app):
        _use_provider(app, configured=True)
        resp = await client.post(
            f"{_PREFIX}/signup", json={"email": "a@example.com", "password": _GOOD_PW}
        )
        assert resp.status_code == 404

    async def test_unconfigured_provider_returns_404(self, client, app, registry_session):
        await _enable_signup(registry_session)
        _use_provider(app, configured=False)
        resp = await client.post(
            f"{_PREFIX}/signup", json={"email": "a@example.com", "password": _GOOD_PW}
        )
        assert resp.status_code == 404

    async def test_creates_pending_account_and_sends(self, client, app, registry_session):
        await _enable_signup(registry_session)
        fake = _use_provider(app)

        resp = await client.post(
            f"{_PREFIX}/signup", json={"email": "New@Example.com", "password": _GOOD_PW}
        )
        assert resp.status_code == 202
        assert "detail" in resp.json()
        assert len(fake.sent) == 1

        user = (await registry_session.execute(
            select(User).where(User.email == "new@example.com")
        )).scalar_one()
        assert user.email_verified_at is None  # pending
        token = (await registry_session.execute(
            select(EmailVerificationToken).where(EmailVerificationToken.user_id == user.id)
        )).scalar_one()
        assert token.used_at is None

    async def test_weak_password_rejected(self, client, app, registry_session):
        await _enable_signup(registry_session)
        _use_provider(app)
        resp = await client.post(
            f"{_PREFIX}/signup", json={"email": "a@example.com", "password": "weak"}
        )
        assert resp.status_code == 422

    async def test_resend_invalidates_prior_token(self, client, app, registry_session):
        await _enable_signup(registry_session)
        fake = _use_provider(app)

        await client.post(
            f"{_PREFIX}/signup", json={"email": "a@example.com", "password": _GOOD_PW}
        )
        await client.post(
            f"{_PREFIX}/signup", json={"email": "a@example.com", "password": "Newpass98765"}
        )
        assert len(fake.sent) == 2

        user = (await registry_session.execute(
            select(User).where(User.email == "a@example.com")
        )).scalar_one()
        tokens = (await registry_session.execute(
            select(EmailVerificationToken).where(EmailVerificationToken.user_id == user.id)
        )).scalars().all()
        unused = [t for t in tokens if t.used_at is None]
        assert len(unused) == 1  # only the latest is live

    async def test_existing_verified_email_is_silent_noop(
        self, client, app, registry_session
    ):
        await _enable_signup(registry_session)
        await _add_verified_user(registry_session, "taken@example.com")
        fake = _use_provider(app)

        resp = await client.post(
            f"{_PREFIX}/signup", json={"email": "taken@example.com", "password": _GOOD_PW}
        )
        assert resp.status_code == 202  # no enumeration
        assert fake.sent == []  # nothing sent for an existing account

    async def test_concurrent_email_race_collapses_to_ack(
        self, client, app, registry_session, monkeypatch
    ):
        # The unique-email race: the losing writer's commit raises IntegrityError.
        # It must collapse to the same generic 202, not surface a 500 (which would
        # both break the endpoint and leak that the email is taken).
        await _enable_signup(registry_session)
        _use_provider(app)

        async def _raise_integrity():
            raise IntegrityError("INSERT", {}, Exception("UNIQUE constraint failed"))

        monkeypatch.setattr(registry_session, "commit", _raise_integrity)

        resp = await client.post(
            f"{_PREFIX}/signup", json={"email": "race@example.com", "password": _GOOD_PW}
        )
        assert resp.status_code == 202
        assert "detail" in resp.json()


# ── /verify-email ───────────────────────────────────────────────────────────


class TestVerifyEmail:
    async def test_verify_activates_and_logs_in(self, client, app, registry_session):
        await _enable_signup(registry_session)
        fake = _use_provider(app)
        await client.post(
            f"{_PREFIX}/signup", json={"email": "a@example.com", "password": _GOOD_PW}
        )
        token = _extract_token(fake.sent[0])

        resp = await client.post(f"{_PREFIX}/verify-email", json={"token": token})
        assert resp.status_code == 200
        assert "access_token" in resp.json()

        user = (await registry_session.execute(
            select(User).where(User.email == "a@example.com")
        )).scalar_one()
        assert user.email_verified_at is not None

    async def test_invalid_token_rejected(self, client, app):
        resp = await client.post(f"{_PREFIX}/verify-email", json={"token": "nope"})
        assert resp.status_code == 400

    async def test_expired_token_rejected(self, client, app, registry_session):
        user = User(
            id=str(uuid.uuid4()),
            email="a@example.com",
            password_hash=hash_password(_GOOD_PW),
            roles=json.dumps(["user"]),
        )
        registry_session.add(user)
        raw = secrets.token_urlsafe(32)
        registry_session.add(EmailVerificationToken(
            user_id=user.id,
            token_hash=hashlib.sha256(raw.encode()).hexdigest(),
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        ))
        await registry_session.commit()

        resp = await client.post(f"{_PREFIX}/verify-email", json={"token": raw})
        assert resp.status_code == 400

    async def test_token_is_single_use(self, client, app, registry_session):
        await _enable_signup(registry_session)
        fake = _use_provider(app)
        await client.post(
            f"{_PREFIX}/signup", json={"email": "a@example.com", "password": _GOOD_PW}
        )
        token = _extract_token(fake.sent[0])

        assert (await client.post(f"{_PREFIX}/verify-email", json={"token": token})).status_code == 200
        assert (await client.post(f"{_PREFIX}/verify-email", json={"token": token})).status_code == 400


# ── login by email ──────────────────────────────────────────────────────────


class TestLoginByEmail:
    async def test_login_with_verified_email(self, client, app, registry_session):
        await _add_verified_user(registry_session, "a@example.com")
        resp = await client.post(
            f"{_PREFIX}/login", json={"username": "a@example.com", "password": _GOOD_PW}
        )
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    async def test_login_email_case_insensitive(self, client, app, registry_session):
        await _add_verified_user(registry_session, "a@example.com")
        resp = await client.post(
            f"{_PREFIX}/login", json={"username": "A@Example.com", "password": _GOOD_PW}
        )
        assert resp.status_code == 200

    async def test_unverified_email_cannot_log_in(self, client, app, registry_session):
        await _enable_signup(registry_session)
        fake = _use_provider(app)
        await client.post(
            f"{_PREFIX}/signup", json={"email": "a@example.com", "password": _GOOD_PW}
        )
        resp = await client.post(
            f"{_PREFIX}/login", json={"username": "a@example.com", "password": _GOOD_PW}
        )
        assert resp.status_code == 401


# ── /request-password-reset ─────────────────────────────────────────────────


class TestRequestPasswordReset:
    async def test_verified_account_gets_link(self, client, app, registry_session):
        user = await _add_verified_user(registry_session, "a@example.com")
        fake = _use_provider(app)

        resp = await client.post(
            f"{_PREFIX}/request-password-reset", json={"email": "a@example.com"}
        )
        assert resp.status_code == 200
        assert len(fake.sent) == 1

        token = (await registry_session.execute(
            select(PasswordResetToken).where(PasswordResetToken.user_id == user.id)
        )).scalar_one()
        assert token.used_at is None

    async def test_unknown_email_is_generic_noop(self, client, app, registry_session):
        fake = _use_provider(app)
        resp = await client.post(
            f"{_PREFIX}/request-password-reset", json={"email": "nobody@example.com"}
        )
        assert resp.status_code == 200  # same generic success
        assert fake.sent == []

    async def test_unconfigured_provider_is_generic_noop(self, client, app, registry_session):
        await _add_verified_user(registry_session, "a@example.com")
        fake = _use_provider(app, configured=False)
        resp = await client.post(
            f"{_PREFIX}/request-password-reset", json={"email": "a@example.com"}
        )
        assert resp.status_code == 200
        assert fake.sent == []


# ── Activation idempotency ──────────────────────────────────────────────────


class TestActivationIdempotency:
    async def test_create_user_profile_is_idempotent(self, isolate_user_dbs):
        # A retry after a partial activation must complete the profile, not
        # duplicate it — this is what makes verify-email recoverable.
        user_id = "idem-user-0000"
        await _create_user_profile(user_id, "Alice")
        await _create_user_profile(user_id, "Alice")

        async with get_user_session_factory(user_id)() as s:
            count = (await s.execute(
                select(func.count()).select_from(Athlete).where(
                    Athlete.global_user_id == user_id
                )
            )).scalar_one()
        assert count == 1
