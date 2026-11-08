"""Unit tests for the clean-up of never-confirmed self-serve signups."""
import itertools
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from backend.app.core.config import settings
from backend.app.models.registry_orm import EmailVerificationToken, User
from backend.app.services import pending_signups
from backend.app.services.pending_signups import (
    PENDING_SIGNUP_TTL,
    run_pending_signup_cleanup,
)

_NOW = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
_counter = itertools.count()


def _pending(session, *, age: timedelta, **overrides) -> User:
    """A signup row as ``POST /signup`` leaves it, with its one-hour token."""
    n = next(_counter)
    created = _NOW - age
    fields = {
        "id": f"pending-{n}",
        "email": f"pending-{n}@example.com",
        "password_hash": "x",
        "created_at": created,
    }
    fields.update(overrides)
    user = User(**fields)
    session.add(user)
    session.add(EmailVerificationToken(
        user_id=user.id,
        token_hash=f"hash-{n}",
        created_at=created,
        expires_at=created + timedelta(hours=1),
    ))
    return user


async def _user_ids(session) -> set[str]:
    return set((await session.execute(select(User.id))).scalars())


async def test_drops_signup_unconfirmed_for_30_days(registry_session):
    old = _pending(registry_session, age=PENDING_SIGNUP_TTL + timedelta(hours=1))
    await registry_session.commit()

    removed = await run_pending_signup_cleanup(registry_session, now=_NOW)

    assert removed == 1
    assert old.id not in await _user_ids(registry_session)
    tokens = (await registry_session.execute(select(EmailVerificationToken))).scalars().all()
    assert tokens == []


async def test_keeps_signup_younger_than_30_days(registry_session):
    young = _pending(registry_session, age=PENDING_SIGNUP_TTL - timedelta(hours=1))
    await registry_session.commit()

    assert await run_pending_signup_cleanup(registry_session, now=_NOW) == 0
    assert young.id in await _user_ids(registry_session)


async def test_resignup_restarts_the_clock(registry_session):
    user = _pending(registry_session, age=timedelta(days=60))
    registry_session.add(EmailVerificationToken(
        user_id=user.id,
        token_hash="resignup",
        created_at=_NOW - timedelta(days=2),
        expires_at=_NOW - timedelta(days=2) + timedelta(hours=1),
    ))
    await registry_session.commit()

    assert await run_pending_signup_cleanup(registry_session, now=_NOW) == 0
    assert user.id in await _user_ids(registry_session)


async def test_never_touches_live_token(registry_session):
    user = _pending(registry_session, age=timedelta(days=60))
    registry_session.add(EmailVerificationToken(
        user_id=user.id,
        token_hash="still-live",
        created_at=_NOW - timedelta(days=40),
        expires_at=_NOW + timedelta(minutes=5),
    ))
    await registry_session.commit()

    assert await run_pending_signup_cleanup(registry_session, now=_NOW) == 0


async def test_keeps_verified_and_invited_accounts(registry_session):
    verified = _pending(
        registry_session, age=timedelta(days=90), email_verified_at=_NOW - timedelta(days=89)
    )
    # Invited/legacy accounts have a username and never verify an address.
    invited = User(
        id="invited", username="invited", password_hash="x",
        created_at=_NOW - timedelta(days=90),
    )
    registry_session.add(invited)
    await registry_session.commit()

    assert await run_pending_signup_cleanup(registry_session, now=_NOW) == 0
    ids = await _user_ids(registry_session)
    assert {verified.id, invited.id} <= ids


async def test_signup_without_any_token_is_dropped(registry_session):
    user = User(
        id="tokenless", email="tokenless@example.com", password_hash="x",
        created_at=_NOW - timedelta(days=45),
    )
    registry_session.add(user)
    await registry_session.commit()

    assert await run_pending_signup_cleanup(registry_session, now=_NOW) == 1
    assert "tokenless" not in await _user_ids(registry_session)


async def test_removes_half_activated_data_dir(registry_session):
    user = _pending(registry_session, age=timedelta(days=45))
    await registry_session.commit()
    data_dir = settings.user_data_dir(user.id)
    data_dir.mkdir(parents=True)
    (data_dir / "user.db").write_bytes(b"")

    assert await run_pending_signup_cleanup(registry_session, now=_NOW) == 1
    assert not data_dir.exists()


async def test_frees_the_address_for_a_new_signup(registry_session):
    user = _pending(registry_session, age=timedelta(days=45), email="reuse@example.com")
    await registry_session.commit()

    await run_pending_signup_cleanup(registry_session, now=_NOW)

    registry_session.add(User(id="fresh", email="reuse@example.com", password_hash="y"))
    await registry_session.commit()
    assert user.id not in await _user_ids(registry_session)


def test_ttl_is_thirty_days():
    assert pending_signups.PENDING_SIGNUP_TTL == timedelta(days=30)
