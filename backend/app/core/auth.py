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

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/teams/{slug}/auth/login")


@dataclass
class TeamContext:
    """Identity + team context extracted from a validated access token."""
    user_id: str    # global user UUID (registry users.id)
    team_id: str    # team UUID
    roles: list[str]

    @property
    def is_admin(self) -> bool:
        return "administrator" in self.roles

    @property
    def is_coach(self) -> bool:
        return "coach" in self.roles


def hash_password(plain: str) -> str:
    return _bcrypt.hashpw(plain.encode(), _bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return _bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(user_id: str, team_id: str, roles: list[str]) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    return jwt.encode(
        {
            "sub": user_id,
            "team_id": team_id,
            "roles": roles,
            "exp": expire,
            "type": "access",
        },
        settings.secret_key,
        algorithm="HS256",
    )


def create_refresh_token(user_id: str, team_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        days=settings.refresh_token_expire_days
    )
    return jwt.encode(
        {"sub": user_id, "team_id": team_id, "exp": expire, "type": "refresh"},
        settings.secret_key,
        algorithm="HS256",
    )


def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.secret_key, algorithms=["HS256"])


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    registry_session: AsyncSession = Depends(get_registry_session),
) -> TeamContext:
    from backend.app.models.registry_orm import User

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
        user_id: str | None = payload.get("sub")
        team_id: str | None = payload.get("team_id")
        if not user_id or not team_id or payload.get("type") != "access":
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    result = await registry_session.execute(
        select(User).where(User.id == user_id, User.deleted_at.is_(None))
    )
    if result.scalar_one_or_none() is None:
        raise credentials_exception
    # Release the pool connection immediately — the user object is no longer
    # needed, but the dependency will keep the session alive until request end.
    # Without this, the session holds the registry pool slot while waiting for
    # the team session, starving concurrent auth checks.
    await registry_session.close()

    roles = payload.get("roles", [])
    if isinstance(roles, str):
        roles = json.loads(roles)

    return TeamContext(user_id=user_id, team_id=team_id, roles=roles)
