"""Personal access tokens — minting, parsing and verification (issue #46).

A PAT is opaque and DB-backed rather than a JWT: a long-lived JWT signed with
``SECRET_KEY`` would need this same lookup table to be revocable, so it would buy
nothing, and ``SECRET_KEY`` already carries three unrelated purposes (the
``access`` and ``refresh`` token types and the OAuth ``state`` claim).

Format::

    okp_{token_id}_{secret}

The ``okp_`` prefix is greppable — by the resolver when routing a bearer value,
by secret scanners, and by a user grepping their own shell history. The embedded
id turns verification into one indexed equality lookup. The secret is 256 bits
from :func:`secrets.token_urlsafe`.

Only ``sha256(secret)`` is persisted, which is the pattern every other non-password
credential here already uses. Deliberately **not** bcrypt: the secret is
high-entropy so there is no brute-force surface to defend, bcrypt would add
~0.27 s to *every API request*, and a bcrypt hash cannot be looked up. Comparison
is :func:`hmac.compare_digest`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.registry_orm import PersonalAccessToken

#: Greppable prefix identifying a personal access token in a bearer header.
TOKEN_PREFIX = "okp_"

#: Lifetimes the UI offers, in days. There is deliberately no "never": the
#: invitation UI offers it and that is right *there* — an unused invite does
#: nothing — but a credential that never dies outlives the integration it was
#: made for, the laptop it was stored on, and usually the memory of creating it.
ALLOWED_LIFETIME_DAYS: tuple[int, ...] = (7, 30, 90, 180, 365)
DEFAULT_LIFETIME_DAYS = 90
#: Hard ceiling, enforced server-side on create so a hand-rolled POST asking for
#: ten years is rejected rather than merely absent from the picker.
MAX_LIFETIME_DAYS = 365

#: How stale ``last_used_at`` may get before it is rewritten. Updating it on
#: every request would amplify writes against a WAL SQLite registry with
#: ``pool_size=3``; the column exists for the UI and for spotting dead tokens,
#: neither of which needs second-level precision.
LAST_USED_REFRESH = timedelta(hours=1)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: Optional[datetime]) -> Optional[datetime]:
    """SQLite hands back naive datetimes; every comparison here is in UTC."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def verify_secret(secret: str, stored_hash: str) -> bool:
    """Timing-safe comparison of a presented secret against the stored hash."""
    return hmac.compare_digest(hash_secret(secret), stored_hash)


def looks_like_pat(raw: str) -> bool:
    return raw.startswith(TOKEN_PREFIX)


def mint_token() -> tuple[str, str, str]:
    """Mint a new credential.

    Returns ``(token_id, raw_token, token_hash)``. The raw token is the only time
    the secret exists outside the caller's hands — it is shown once at creation
    and never again.
    """
    token_id = str(uuid.uuid4())
    secret = secrets.token_urlsafe(32)
    return token_id, f"{TOKEN_PREFIX}{token_id}_{secret}", hash_secret(secret)


def parse_token(raw: str) -> Optional[tuple[str, str]]:
    """Split a presented token into ``(token_id, secret)``.

    Returns ``None`` for anything that is not shaped like a PAT. The secret's
    urlsafe alphabet includes ``_``, so the split is capped at two — the id is a
    UUID and never contains one, which keeps the parse unambiguous.
    """
    if not looks_like_pat(raw):
        return None
    parts = raw.split("_", 2)
    if len(parts) != 3:
        return None
    _, token_id, secret = parts
    if not token_id or not secret:
        return None
    return token_id, secret


def scopes_of(token: PersonalAccessToken) -> list[str]:
    try:
        loaded = json.loads(token.scopes) if token.scopes else []
    except (TypeError, ValueError):
        return []
    # Anything that isn't a list of strings grants nothing. A scope list this
    # code cannot read must deny, never widen.
    if not isinstance(loaded, list):
        return []
    return [s for s in loaded if isinstance(s, str)]


def status_of(token: PersonalAccessToken, now: Optional[datetime] = None) -> str:
    """``"active"``, ``"revoked"`` or ``"expired"`` — in that order of precedence.

    Revocation wins over expiry so a token withdrawn before its time still reads
    as a deliberate act rather than as one that simply ran out.
    """
    now = now or _now()
    if token.revoked_at is not None:
        return "revoked"
    expires_at = _aware(token.expires_at)
    if expires_at is not None and expires_at <= now:
        return "expired"
    return "active"


def is_active(token: PersonalAccessToken, now: Optional[datetime] = None) -> bool:
    return status_of(token, now) == "active"


def expires_at_for(lifetime_days: int, now: Optional[datetime] = None) -> datetime:
    """Expiry for a requested lifetime, or ``ValueError`` past the ceiling."""
    if lifetime_days < 1 or lifetime_days > MAX_LIFETIME_DAYS:
        raise ValueError(
            f"Lifetime must be between 1 and {MAX_LIFETIME_DAYS} days."
        )
    return (now or _now()) + timedelta(days=lifetime_days)


async def load_by_id(
    session: AsyncSession, token_id: str
) -> Optional[PersonalAccessToken]:
    result = await session.execute(
        select(PersonalAccessToken).where(PersonalAccessToken.id == token_id)
    )
    return result.scalar_one_or_none()


async def touch_last_used(
    session: AsyncSession,
    token: PersonalAccessToken,
    now: Optional[datetime] = None,
) -> bool:
    """Refresh ``last_used_at``, but only when it is more than an hour stale.

    Returns whether a write happened, which is what the caller's tests assert on.
    """
    now = now or _now()
    last_used = _aware(token.last_used_at)
    if last_used is not None and now - last_used < LAST_USED_REFRESH:
        return False
    token.last_used_at = now
    await session.commit()
    return True


async def revoke_all_for_user(
    session: AsyncSession, user_id: str, now: Optional[datetime] = None
) -> int:
    """Revoke every live token a user holds. Returns how many were withdrawn.

    Called on password reset: whatever prompted the reset — a suspected
    compromise most of the time — applies to the credentials the account handed
    out just as much as to the password itself.
    """
    now = now or _now()
    result = await session.execute(
        update(PersonalAccessToken)
        .where(
            PersonalAccessToken.user_id == user_id,
            PersonalAccessToken.revoked_at.is_(None),
        )
        .values(revoked_at=now)
    )
    return int(result.rowcount or 0)
