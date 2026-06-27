import hashlib
import json
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from jose import JWTError
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import (
    TeamContext,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    hash_password,
    verify_password,
)
from backend.app.core.config import settings
from backend.app.core.limiter import limiter
from backend.app.db.registry import get_registry_session
from backend.app.db.team_session import init_team_db, get_team_session_factory
from backend.app.db.user_session import delete_user_db
from backend.app.models.registry_orm import (
    Invitation,
    PasswordResetToken,
    ProviderConnection,
    Team,
    TeamMembership,
    User,
)
from backend.app.models.team_orm import Athlete
from backend.app.schemas.auth import (
    DeleteAccountRequest,
    LoginRequest,
    RegisterRequest,
    ResetPasswordRequest,
    TokenResponse,
)
from backend.app.services import notifications
from backend.app.services.providers.registry import PROVIDERS

log = logging.getLogger(__name__)

router = APIRouter(prefix="/teams/{slug}/auth", tags=["auth"])

_COOKIE_NAME = "refresh_token"
_COOKIE_MAX_AGE = settings.refresh_token_expire_days * 24 * 60 * 60


def _cookie_path(slug: str) -> str:
    return f"/api/teams/{slug}/auth"


def _set_refresh_cookie(response: Response, slug: str, token: str) -> None:
    secure = settings.frontend_url.startswith("https://")
    response.set_cookie(
        key=_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=secure,
        samesite="lax",
        max_age=_COOKIE_MAX_AGE,
        path=_cookie_path(slug),
    )


def _clear_refresh_cookie(response: Response, slug: str) -> None:
    response.delete_cookie(key=_COOKIE_NAME, path=_cookie_path(slug))


async def _resolve_team(slug: str, session: AsyncSession) -> Team:
    result = await session.execute(select(Team).where(Team.slug == slug))
    team = result.scalar_one_or_none()
    if team is None:
        raise HTTPException(status_code=404, detail="Team not found")
    return team


@router.post("/login", response_model=TokenResponse)
@limiter.limit("20/minute")
async def login(
    request: Request,
    slug: str,
    body: LoginRequest,
    response: Response,
    session: AsyncSession = Depends(get_registry_session),
):
    team = await _resolve_team(slug, session)
    if team.status == "pending":
        raise HTTPException(status_code=403, detail="Team pending approval")
    if team.status == "rejected":
        raise HTTPException(status_code=403, detail="Team access revoked")

    result = await session.execute(
        select(User).where(User.username == body.username, User.deleted_at.is_(None))
    )
    user = result.scalar_one_or_none()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    membership_result = await session.execute(
        select(TeamMembership).where(
            TeamMembership.team_id == team.id,
            TeamMembership.user_id == user.id,
        )
    )
    membership = membership_result.scalar_one_or_none()
    if membership is None:
        raise HTTPException(status_code=403, detail="Not a member of this team")

    roles = json.loads(membership.roles)
    _set_refresh_cookie(response, slug, create_refresh_token(user.id, team.id))
    return TokenResponse(access_token=create_access_token(user.id, team.id, roles))


@router.post("/register", response_model=TokenResponse, status_code=201)
@limiter.limit("10/hour")
async def register(
    request: Request,
    slug: str,
    body: RegisterRequest,
    response: Response,
    session: AsyncSession = Depends(get_registry_session),
):
    team = await _resolve_team(slug, session)

    # Validate invite token
    token_hash = hashlib.sha256(body.invite_token.encode()).hexdigest()
    inv_result = await session.execute(
        select(Invitation).where(
            Invitation.team_id == team.id,
            Invitation.token_hash == token_hash,
            Invitation.used_at.is_(None),
        )
    )
    invitation = inv_result.scalar_one_or_none()
    if invitation is None:
        raise HTTPException(status_code=400, detail="Invalid or expired invite token")

    now = datetime.now(timezone.utc)
    if invitation.expires_at is not None:
        expires_at = invitation.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= now:
            raise HTTPException(status_code=400, detail="Invite token has expired")

    # Create or reuse the global user
    existing_user = await session.execute(
        select(User).where(User.username == body.username)
    )
    user = existing_user.scalar_one_or_none()
    if user is not None:
        if user.deleted_at is not None:
            raise HTTPException(status_code=400, detail="Username not available")
        # User already exists globally — check they're not already in this team
        existing_mb = await session.execute(
            select(TeamMembership).where(
                TeamMembership.team_id == team.id,
                TeamMembership.user_id == user.id,
            )
        )
        if existing_mb.scalar_one_or_none() is not None:
            raise HTTPException(status_code=400, detail="Already a member of this team")
    else:
        user = User(
            id=str(uuid.uuid4()),
            username=body.username,
            password_hash=hash_password(body.password),
        )
        session.add(user)
        await session.flush()

    roles = json.loads(invitation.roles)
    membership = TeamMembership(
        team_id=team.id,
        user_id=user.id,
        roles=json.dumps(roles),
    )
    session.add(membership)

    invitation.used_at = now
    invitation.used_by_user_id = user.id
    await session.commit()

    # Create athlete profile in the team DB
    await init_team_db(team.id)
    from backend.app.models.team_orm import Athlete
    async with get_team_session_factory(team.id)() as team_session:
        athlete = Athlete(
            id=str(uuid.uuid4()),
            global_user_id=user.id,
            name=body.display_name or None,
            ftp_tests=[],
        )
        team_session.add(athlete)
        await team_session.commit()

    await notifications.notify_team_admins(
        session,
        team.id,
        notifications.INVITE_USED,
        {
            "username": user.username,
            "display_name": body.display_name or None,
            "team_name": team.name,
            "team_slug": team.slug,
        },
    )

    _set_refresh_cookie(response, slug, create_refresh_token(user.id, team.id))
    return TokenResponse(access_token=create_access_token(user.id, team.id, roles))


@router.post("/refresh", response_model=TokenResponse)
@limiter.limit("60/minute")
async def refresh(
    request: Request,
    slug: str,
    response: Response,
    refresh_token: str | None = Cookie(default=None, alias=_COOKIE_NAME),
    session: AsyncSession = Depends(get_registry_session),
):
    if not refresh_token:
        raise HTTPException(status_code=401, detail="No refresh token")
    try:
        payload = decode_token(refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user_id: str | None = payload.get("sub")
        team_id: str | None = payload.get("team_id")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    result = await session.execute(
        select(User).where(User.id == user_id, User.deleted_at.is_(None))
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    membership_result = await session.execute(
        select(TeamMembership).where(
            TeamMembership.team_id == team_id,
            TeamMembership.user_id == user_id,
        )
    )
    membership = membership_result.scalar_one_or_none()
    if membership is None:
        raise HTTPException(status_code=403, detail="Not a member of this team")

    roles = json.loads(membership.roles)
    _set_refresh_cookie(response, slug, create_refresh_token(user_id, team_id))
    return TokenResponse(access_token=create_access_token(user_id, team_id, roles))


@router.post("/logout", status_code=204)
async def logout(slug: str, response: Response):
    _clear_refresh_cookie(response, slug)


@router.delete("/account", status_code=204)
async def delete_account(
    slug: str,
    body: DeleteAccountRequest,
    response: Response,
    ctx: TeamContext = Depends(get_current_user),
    session: AsyncSession = Depends(get_registry_session),
):
    result = await session.execute(
        select(User).where(User.id == ctx.user_id, User.deleted_at.is_(None))
    )
    user = result.scalar_one_or_none()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid password")

    # Revoke all provider connections (best-effort)
    conn_result = await session.execute(
        select(ProviderConnection).where(ProviderConnection.user_id == ctx.user_id)
    )
    for conn in conn_result.scalars().all():
        if conn.access_token and conn.provider in PROVIDERS:
            try:
                await PROVIDERS[conn.provider].deauthorize(conn.access_token)  # type: ignore[call-arg]
            except Exception:
                pass

    # Delete athlete data from every team the user belongs to
    mb_result = await session.execute(
        select(TeamMembership).where(TeamMembership.user_id == ctx.user_id)
    )
    for membership in mb_result.scalars().all():
        try:
            async with get_team_session_factory(membership.team_id)() as team_session:
                athlete_result = await team_session.execute(
                    select(Athlete).where(Athlete.global_user_id == ctx.user_id)
                )
                athlete = athlete_result.scalar_one_or_none()
                if athlete is None:
                    continue
                if athlete.avatar_path:
                    Path(athlete.avatar_path).unlink(missing_ok=True)
                await team_session.delete(athlete)
                await team_session.commit()
        except Exception:
            log.exception(
                "Failed to delete athlete data for user %s in team %s",
                ctx.user_id,
                membership.team_id,
            )

    # Remove all team memberships explicitly before deleting the user
    await session.execute(delete(TeamMembership).where(TeamMembership.user_id == ctx.user_id))
    # Hard-delete the user; cascades to provider connections and reset tokens
    await session.delete(user)
    await session.commit()

    # Remove the user's per-user DB (message inbox, etc.) so nothing is orphaned.
    try:
        await delete_user_db(ctx.user_id)
    except Exception:
        log.exception("Failed to delete per-user DB for user %s", ctx.user_id)

    _clear_refresh_cookie(response, slug)


@router.post("/reset-password", status_code=204)
@limiter.limit("10/hour")
async def reset_password(
    request: Request,
    slug: str,
    body: ResetPasswordRequest,
    session: AsyncSession = Depends(get_registry_session),
):
    token_hash = hashlib.sha256(body.token.encode()).hexdigest()
    result = await session.execute(
        select(PasswordResetToken).where(PasswordResetToken.token_hash == token_hash)
    )
    token_row = result.scalar_one_or_none()

    now = datetime.now(timezone.utc)
    if token_row is None or token_row.used_at is not None:
        raise HTTPException(status_code=400, detail="Invalid or expired token")
    expires_at = token_row.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at <= now:
        raise HTTPException(status_code=400, detail="Invalid or expired token")

    user_result = await session.execute(
        select(User).where(User.id == token_row.user_id, User.deleted_at.is_(None))
    )
    user = user_result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=400, detail="Invalid or expired token")

    user.password_hash = hash_password(body.new_password)
    token_row.used_at = now
    await session.commit()
