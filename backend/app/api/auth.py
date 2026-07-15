import hashlib
import json
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone

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
    EmailVerificationToken,
    InstanceSettings,
    Invitation,
    PasswordResetToken,
    ProviderConnection,
    User,
)
from backend.app.models.user_orm import Athlete
from backend.app.schemas.auth import (
    DeleteAccountRequest,
    LoginRequest,
    MessageResponse,
    RegisterRequest,
    RequestPasswordResetRequest,
    ResetPasswordRequest,
    SignupRequest,
    TokenResponse,
    VerifyEmailRequest,
)
from backend.app.services import notifications
from backend.app.services.email import (
    EmailError,
    EmailProvider,
    get_email_provider,
    send_password_reset_email,
    send_verification_email,
)
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


def get_email_provider_dep() -> EmailProvider:
    """Injectable wrapper around the configured provider (overridable in tests)."""
    return get_email_provider()


async def _create_user_profile(user_id: str, display_name: str | None) -> None:
    """Bootstrap a newly activated account's per-user DB + athlete profile.

    Shared by invite ``register`` and self-serve ``verify_email`` so both
    activate an account identically.
    """
    await init_user_db(user_id)
    async with get_user_session_factory(user_id)() as user_session:
        athlete = Athlete(
            id=str(uuid.uuid4()),
            global_user_id=user_id,
            name=display_name or None,
            ftp_tests=[],
        )
        user_session.add(athlete)
        await user_session.commit()


async def _self_signup_enabled(session: AsyncSession, provider: EmailProvider) -> bool:
    """Whether self-serve signup is currently offered (admin toggle + email)."""
    result = await session.execute(select(InstanceSettings).limit(1))
    instance = result.scalar_one_or_none()
    return bool(instance and instance.allow_self_signup) and provider.is_configured


@router.post("/login", response_model=TokenResponse, operation_id="login", summary="Log in")
@limiter.limit("20/minute")
async def login(
    request: Request,
    body: LoginRequest,
    response: Response,
    session: AsyncSession = Depends(get_registry_session),
):
    # Accept either a username (invited/legacy accounts) or a verified email
    # address (self-serve signup accounts) as the login identifier.
    identifier = body.username
    result = await session.execute(
        select(User).where(User.username == identifier, User.deleted_at.is_(None))
    )
    user = result.scalar_one_or_none()
    if user is None:
        result = await session.execute(
            select(User).where(
                User.email == identifier.lower(),
                User.email_verified_at.is_not(None),
                User.deleted_at.is_(None),
            )
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
    await _create_user_profile(user.id, body.display_name)

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


# ── Self-serve signup + email verification (issue #15) ──────────────────────

_SIGNUP_ACK = "If self-serve signup is available, check your inbox to confirm your email."
_RESET_ACK = "If an account exists for that email, a password-reset link has been sent."


@router.post("/signup", response_model=MessageResponse, status_code=202,
             operation_id="signup", summary="Sign up with an email address")
@limiter.limit("10/hour")
async def signup(
    request: Request,
    body: SignupRequest,
    provider: EmailProvider = Depends(get_email_provider_dep),
    session: AsyncSession = Depends(get_registry_session),
):
    """Create a pending account and email a verification link.

    Guarded by the ``allow_self_signup`` admin toggle and a configured email
    provider. Always returns the same generic acknowledgement (no account
    enumeration): re-signing up an unverified account resends the link with the
    new password; an already-verified email is a silent no-op.
    """
    if not await _self_signup_enabled(session, provider):
        raise HTTPException(status_code=404, detail="Self-serve signup is not available")

    ack = MessageResponse(detail=_SIGNUP_ACK)
    email = str(body.email).lower()
    now = datetime.now(timezone.utc)

    existing = await session.execute(select(User).where(User.email == email))
    user = existing.scalar_one_or_none()
    if user is not None and user.email_verified_at is not None:
        # Already a real account — say nothing that reveals it.
        return ack

    if user is None:
        user = User(
            id=str(uuid.uuid4()),
            email=email,
            password_hash=hash_password(body.password),
            roles=json.dumps(["user"]),
        )
        session.add(user)
        await session.flush()
    else:
        # Pending re-signup: let the latest attempt set the password.
        user.password_hash = hash_password(body.password)

    prior = await session.execute(
        select(EmailVerificationToken).where(
            EmailVerificationToken.user_id == user.id,
            EmailVerificationToken.used_at.is_(None),
        )
    )
    for token_row in prior.scalars():
        token_row.used_at = now

    raw_token = secrets.token_urlsafe(32)
    session.add(EmailVerificationToken(
        user_id=user.id,
        token_hash=hashlib.sha256(raw_token.encode()).hexdigest(),
        expires_at=now + timedelta(hours=1),
    ))
    await session.commit()

    verify_url = f"{settings.frontend_url}/verify-email?token={raw_token}"
    try:
        await send_verification_email(provider, to=email, action_url=verify_url)
    except EmailError:
        # Delivery failed after a pending account was created; the user can retry
        # signup to resend. Don't leak the failure into the generic response.
        log.exception("Failed to send verification email to a signup address")
    return ack


@router.post("/verify-email", response_model=TokenResponse,
             operation_id="verifyEmail", summary="Verify email and activate account")
@limiter.limit("20/hour")
async def verify_email(
    request: Request,
    body: VerifyEmailRequest,
    response: Response,
    session: AsyncSession = Depends(get_registry_session),
):
    """Consume a verification token, mark the email verified, and activate the
    account (creating its per-user DB + athlete profile), then log the user in."""
    token_hash = hashlib.sha256(body.token.encode()).hexdigest()
    result = await session.execute(
        select(EmailVerificationToken).where(
            EmailVerificationToken.token_hash == token_hash
        )
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

    already_active = user.email_verified_at is not None
    if not already_active:
        user.email_verified_at = now
    token_row.used_at = now
    await session.commit()

    if not already_active:
        await _create_user_profile(user.id, None)

    roles = _roles_of(user)
    _set_refresh_cookie(response, create_refresh_token(user.id))
    return TokenResponse(access_token=create_access_token(user.id, roles))


@router.post("/request-password-reset", response_model=MessageResponse,
             operation_id="requestPasswordReset",
             summary="Email a password-reset link")
@limiter.limit("10/hour")
async def request_password_reset(
    request: Request,
    body: RequestPasswordResetRequest,
    provider: EmailProvider = Depends(get_email_provider_dep),
    session: AsyncSession = Depends(get_registry_session),
):
    """Email a reset link to a verified account. Always returns the same generic
    acknowledgement (no account enumeration); a no-op when email is unconfigured
    or no verified account matches."""
    ack = MessageResponse(detail=_RESET_ACK)
    if not provider.is_configured:
        return ack

    email = str(body.email).lower()
    now = datetime.now(timezone.utc)
    result = await session.execute(
        select(User).where(
            User.email == email,
            User.email_verified_at.is_not(None),
            User.deleted_at.is_(None),
        )
    )
    user = result.scalar_one_or_none()
    if user is None:
        return ack

    existing = await session.execute(
        select(PasswordResetToken).where(
            PasswordResetToken.user_id == user.id,
            PasswordResetToken.used_at.is_(None),
        )
    )
    for token_row in existing.scalars():
        token_row.used_at = now

    raw_token = secrets.token_urlsafe(32)
    session.add(PasswordResetToken(
        user_id=user.id,
        token_hash=hashlib.sha256(raw_token.encode()).hexdigest(),
        expires_at=now + timedelta(hours=1),
    ))
    await session.commit()

    reset_url = f"{settings.frontend_url}/reset-password?token={raw_token}"
    try:
        await send_password_reset_email(provider, to=email, action_url=reset_url)
    except EmailError:
        log.exception("Failed to send password-reset email")
    return ack
