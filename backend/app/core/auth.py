from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import bcrypt as _bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.db.registry import get_registry_session

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


@dataclass
class UserContext:
    """Identity extracted from a validated access token.

    The instance is single-tenant: there is no team. Every user's data lives in
    their own per-user DB, addressed by ``user_id``.
    """
    user_id: str    # global user UUID (registry users.id)
    roles: list[str]

    @property
    def is_admin(self) -> bool:
        return "administrator" in self.roles


def hash_password(plain: str) -> str:
    return _bcrypt.hashpw(plain.encode(), _bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return _bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(user_id: str, roles: list[str]) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    return jwt.encode(
        {
            "sub": user_id,
            "roles": roles,
            "exp": expire,
            "type": "access",
        },
        settings.secret_key,
        algorithm="HS256",
    )


def create_refresh_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        days=settings.refresh_token_expire_days
    )
    return jwt.encode(
        {"sub": user_id, "exp": expire, "type": "refresh"},
        settings.secret_key,
        algorithm="HS256",
    )


def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.secret_key, algorithms=["HS256"])


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    registry_session: AsyncSession = Depends(get_registry_session),
) -> UserContext:
    from backend.app.models.registry_orm import User

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
        user_id: str | None = payload.get("sub")
        if not user_id or payload.get("type") != "access":
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    result = await registry_session.execute(
        select(User).where(User.id == user_id, User.deleted_at.is_(None))
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise credentials_exception
    # Prefer the authoritative roles from the registry row over the token claim,
    # so a role change takes effect without waiting for token expiry.
    try:
        roles = json.loads(user.roles) if user.roles else []
    except (TypeError, ValueError):
        roles = []
    # Release the pool connection immediately — the user object is no longer
    # needed, but the dependency would otherwise keep the session (and its pool
    # slot) alive until request end while the per-user session is in use.
    await registry_session.close()

    return UserContext(user_id=user_id, roles=roles)
