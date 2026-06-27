import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import (
    UserContext,
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
from backend.app.db.user_session import delete_user_db, get_user_session_factory, init_user_db
from backend.app.models.registry_orm import (
    Invitation,
    PasswordResetToken,
    ProviderConnection,
    User,
)
from backend.app.models.user_orm import Athlete
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

router = APIRouter(prefix="/auth", tags=["auth"])

_COOKIE_NAME = "refresh_token"
_COOKIE_MAX_AGE = settings.refresh_token_expire_days * 24 * 60 * 60
_COOKIE_PATH = "/api/auth"


def _set_refresh_cookie(response: Response, token: str) -> None:
    secure = settings.frontend_url.startswith("https://")
    response.set_cookie(
        key=_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=secure,
        samesite="lax",
        max_age=_COOKIE_MAX_AGE,
        path=_COOKIE_PATH,
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(key=_COOKIE_NAME, path=_COOKIE_PATH)


def _roles_of(user: User) -> list[str]:
    try:
        return json.loads(user.roles) if user.roles else []
    except (TypeError, ValueError):
        return []


@router.post("/login", response_model=TokenResponse, operation_id="login", summary="Log in")
@limiter.limit("20/minute")
async def login(
    request: Request,
    body: LoginRequest,
    response: Response,
    session: AsyncSession = Depends(get_registry_session),
):
    result = await session.execute(
        select(User).where(User.username == body.username, User.deleted_at.is_(None))
    )
    user = result.scalar_one_or_none()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    roles = _roles_of(user)
    _set_refresh_cookie(response, create_refresh_token(user.id))
    return TokenResponse(access_token=create_access_token(user.id, roles))


@router.post("/register", response_model=TokenResponse, status_code=201,
             operation_id="register", summary="Register with an invite token")
@limiter.limit("10/hour")
async def register(
    request: Request,
    body: RegisterRequest,
    response: Response,
    session: AsyncSession = Depends(get_registry_session),
):
    # Validate the instance-wide invite token
    token_hash = hashlib.sha256(body.invite_token.encode()).hexdigest()
    inv_result = await session.execute(
        select(Invitation).where(
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

    existing_user = await session.execute(
        select(User).where(User.username == body.username)
    )
    if existing_user.scalar_one_or_none() is not None:
        raise HTTPException(status_code=400, detail="Username not available")

    roles = json.loads(invitation.roles)
    user = User(
        id=str(uuid.uuid4()),
        username=body.username,
        password_hash=hash_password(body.password),
        roles=json.dumps(roles),
    )
    session.add(user)
    await session.flush()

    invitation.used_at = now
    invitation.used_by_user_id = user.id
    await session.commit()

    # Create the athlete profile in the user's own DB
    await init_user_db(user.id)
    async with get_user_session_factory(user.id)() as user_session:
        athlete = Athlete(
            id=str(uuid.uuid4()),
            global_user_id=user.id,
            name=body.display_name or None,
            ftp_tests=[],
        )
        user_session.add(athlete)
        await user_session.commit()

    await notifications.notify_admins(
        session,
        notifications.INVITE_USED,
        {
            "username": user.username,
            "display_name": body.display_name or None,
        },
    )

    _set_refresh_cookie(response, create_refresh_token(user.id))
    return TokenResponse(access_token=create_access_token(user.id, roles))


@router.post("/refresh", response_model=TokenResponse,
             operation_id="refreshToken", summary="Refresh access token")
@limiter.limit("60/minute")
async def refresh(
    request: Request,
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
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    result = await session.execute(
        select(User).where(User.id == user_id, User.deleted_at.is_(None))
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    roles = _roles_of(user)
    _set_refresh_cookie(response, create_refresh_token(user.id))
    return TokenResponse(access_token=create_access_token(user.id, roles))


@router.post("/logout", status_code=204, operation_id="logout", summary="Log out")
async def logout(response: Response):
    _clear_refresh_cookie(response)


@router.delete("/account", status_code=204,
               operation_id="deleteAccount", summary="Delete the current account")
async def delete_account(
    body: DeleteAccountRequest,
    response: Response,
    ctx: UserContext = Depends(get_current_user),
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

    # Hard-delete the user; cascades to provider connections and reset tokens
    await session.delete(user)
    await session.commit()

    # Remove the user's per-user DB (athlete, all training data, inbox) entirely.
    try:
        await delete_user_db(ctx.user_id)
    except Exception:
        log.exception("Failed to delete per-user DB for user %s", ctx.user_id)

    _clear_refresh_cookie(response)


@router.post("/reset-password", status_code=204,
             operation_id="resetPassword", summary="Reset password with a token")
@limiter.limit("10/hour")
async def reset_password(
    request: Request,
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
