import hashlib
import json
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Cookie,
    Depends,
    HTTPException,
    Request,
    Response,
)
from jose import JWTError
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import (
    UserContext,
    create_access_token,
    create_refresh_token,
    decode_token,
    dummy_password_hash,
    get_current_user,
    hash_password_async,
    invalidate_sessions,
    token_version_matches,
    verify_password_async,
)
from backend.app.core.config import settings
from backend.app.core.limiter import limiter
from backend.app.core.scopes import pat_forbidden
from backend.app.db.registry import get_registry_session
from backend.app.db.user_session import delete_user_db, get_user_session_factory, init_user_db
from backend.app.models.registry_orm import (
    EmailChangeToken,
    EmailVerificationToken,
    InstanceSettings,
    Invitation,
    PasswordResetToken,
    ProviderConnection,
    User,
)
from backend.app.models.user_orm import Athlete
from backend.app.schemas.auth import (
    AccountResponse,
    ChangeEmailRequest,
    ConfirmEmailChangeRequest,
    DeleteAccountRequest,
    EmailChangeConfirmResponse,
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
    send_email_change_email,
    send_email_change_authorisation,
    send_password_reset_email,
    send_verification_email,
)
from backend.app.services.personal_access_tokens import revoke_all_for_user
from backend.app.services.providers.registry import PROVIDERS

log = logging.getLogger(__name__)


router = APIRouter(prefix="/auth", tags=["auth"], dependencies=[pat_forbidden()])

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
    activate an account identically. Idempotent: creating the DB is a no-op if it
    exists, and the athlete row is only inserted when absent, so a retry after a
    partial activation completes it rather than duplicating the profile.
    """
    await init_user_db(user_id)
    async with get_user_session_factory(user_id)() as user_session:
        existing = await user_session.execute(
            select(Athlete).where(Athlete.global_user_id == user_id)
        )
        if existing.scalar_one_or_none() is not None:
            return
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

    # Verify unconditionally, against a fixed dummy hash when nothing matched.
    # Short-circuiting on `user is None` let an unknown identifier skip bcrypt
    # and answer 66× faster than a known one, which is a reliable account
    # oracle and undoes what signup and password-reset go out of their way to
    # hide (#102, F-06). The dummy hashes a value nobody can supply, so the
    # comparison always fails — `user is None` is still checked below, so a
    # freak match could not authenticate nobody.
    password_hash = user.password_hash if user is not None else dummy_password_hash()
    password_ok = await verify_password_async(body.password, password_hash)
    if user is None or not password_ok:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    roles = _roles_of(user)
    _set_refresh_cookie(response, create_refresh_token(user.id, token_version=user.token_version))
    return TokenResponse(
        access_token=create_access_token(user.id, roles, token_version=user.token_version)
    )


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
        password_hash=await hash_password_async(body.password),
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

    _set_refresh_cookie(response, create_refresh_token(user.id, token_version=user.token_version))
    return TokenResponse(
        access_token=create_access_token(user.id, roles, token_version=user.token_version)
    )


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
    # The refresh cookie is the long-lived half — 30 days by default, and it
    # mints fresh access tokens the whole time. Checking the generation here is
    # what actually ends a session; refusing only the access token would buy an
    # hour and hand the holder a new one (F-04).
    if not token_version_matches(payload, user):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    roles = _roles_of(user)
    _set_refresh_cookie(response, create_refresh_token(user.id, token_version=user.token_version))
    return TokenResponse(
        access_token=create_access_token(user.id, roles, token_version=user.token_version)
    )


@router.post("/logout", status_code=204, operation_id="logout", summary="Log out")
async def logout(response: Response):
    # Clears this browser's cookie and nothing else. A session JWT carries no
    # per-session identity, so there is no way to retire one token without
    # retiring them all — which is what /logout-all is for. Ending a single
    # stolen session without disturbing the others would need a session id per
    # token and somewhere to record it.
    _clear_refresh_cookie(response)


@router.post("/logout-all", status_code=204,
             operation_id="logoutAll", summary="Sign out of every device")
async def logout_all(
    response: Response,
    ctx: UserContext = Depends(get_current_user),
    session: AsyncSession = Depends(get_registry_session),
):
    """End every session this account has open, including the calling one.

    The counterpart to a password reset for someone who is still signed in and
    wants their other sessions gone — a shared computer they forgot to sign out
    of, a stolen phone — without changing their password to get it. Every
    access token and refresh cookie the account holds stops working
    immediately; the caller signs in again like anyone else.
    """
    result = await session.execute(
        select(User).where(User.id == ctx.user_id, User.deleted_at.is_(None))
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    invalidate_sessions(user)
    await session.commit()
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
    if user is None or not await verify_password_async(body.password, user.password_hash):
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

    user.password_hash = await hash_password_async(body.new_password)
    token_row.used_at = now
    # Whatever prompted the reset — a suspected compromise, most of the time —
    # applies to the credentials this account handed out just as much as to the
    # password itself, so every live personal access token goes with it (#46).
    await revoke_all_for_user(session, user.id, now)
    # And to the sessions, for the same reason. Without this the reset changed
    # the password while the holder of a stolen access token kept using it, and
    # their refresh cookie kept minting replacements for another 30 days —
    # every token control green, the account still theirs (#102, F-04).
    invalidate_sessions(user)
    # And to any change of address already in flight (issue #62). Revoking the
    # credentials but leaving that armed is not a recovery: an attacker holding
    # the password can request a move to their own address and confirm the new
    # side before the victim ever notices, and the old-side approval then sits
    # live in the victim's inbox for the rest of its 24 hours. The message that
    # put it there is the one telling them to change their password — and with
    # no authenticated change-password endpoint, *this* is that flow. So a reset
    # that spared it would hand the account over on the next curious click, by
    # way of the very advice meant to prevent it.
    #
    # The invariant: recovering an account withdraws every credential it issued
    # **and** every identity change standing against it.
    #
    # Not extended to /logout-all, which costs no credential and means "sign my
    # other devices out" — silently cancelling a change the user is halfway
    # through would be a surprise, not a protection.
    pending_changes = await session.execute(
        select(EmailChangeToken).where(
            EmailChangeToken.user_id == user.id,
            EmailChangeToken.used_at.is_(None),
        )
    )
    for change_row in pending_changes.scalars():
        change_row.used_at = now
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

    try:
        if user is None:
            user = User(
                id=str(uuid.uuid4()),
                email=email,
                password_hash=await hash_password_async(body.password),
                roles=json.dumps(["user"]),
            )
            session.add(user)
            await session.flush()
        else:
            # Pending re-signup: let the latest attempt set the password.
            user.password_hash = await hash_password_async(body.password)

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
    except IntegrityError:
        # Concurrent signup for the same email: the unique constraint fires on the
        # losing writer. Collapse to the generic ack so the response stays uniform
        # (no enumeration) — the winning request already sent a verification email.
        await session.rollback()
        return ack

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

    # Provision the per-user DB + profile *before* committing the activation, so a
    # failure here leaves the token unspent and the account unverified — the user
    # can simply retry the same link (within its 1h window) rather than being
    # stranded verified-but-DB-less. _create_user_profile is idempotent, so the
    # retry completes activation instead of duplicating the profile.
    already_active = user.email_verified_at is not None
    await _create_user_profile(user.id, None)

    if not already_active:
        user.email_verified_at = now
    token_row.used_at = now
    await session.commit()

    roles = _roles_of(user)
    _set_refresh_cookie(response, create_refresh_token(user.id, token_version=user.token_version))
    return TokenResponse(
        access_token=create_access_token(user.id, roles, token_version=user.token_version)
    )


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


# ── Changing (or setting) the account's email address (issue #62) ────────────

_CHANGE_EMAIL_ACK = (
    "If that address can be used, check its inbox to confirm the change."
)


async def _live_email_change(
    session: AsyncSession, user_id: str, now: datetime
) -> EmailChangeToken | None:
    """The caller's outstanding change request, if one is still openable.

    Unused and unexpired — an expired row is not a pending change, it is
    litter, and reporting it would leave the account looking stuck at an
    address it never moved to.
    """
    result = await session.execute(
        select(EmailChangeToken).where(
            EmailChangeToken.user_id == user_id,
            EmailChangeToken.used_at.is_(None),
        ).order_by(EmailChangeToken.created_at.desc())
    )
    for row in result.scalars():
        expires_at = row.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at > now:
            return row
    return None


async def _dead_signup_stub(
    session: AsyncSession, email: str, user_id: str
) -> User | None:
    """Another account holding this address that never finished signing up.

    A self-serve signup writes the user row *before* the address is confirmed,
    so an abandoned attempt leaves a row squatting on it with
    ``email_verified_at`` NULL and a verification token that has long expired.
    :func:`signup` treats such a row as reusable — it resets the password and
    mails a fresh link — so refusing to let a change claim the same address
    would make this flow stricter than the one that created the obstruction, and
    permanently: nothing expires it, and the uniform acknowledgement means the
    user sees no reason why their link never arrives. With self-serve signup on,
    that also makes "sign up as someone and never verify" a way to deny them an
    address for good.

    A row still holding a *live* verification token is a signup in progress, not
    an abandoned one, and keeps the address.
    """
    result = await session.execute(
        select(User).where(
            User.email == email,
            User.id != user_id,
            User.email_verified_at.is_(None),
        ).limit(1)
    )
    stub = result.scalar_one_or_none()
    if stub is None:
        return None

    live = await session.execute(
        select(EmailVerificationToken.id).where(
            EmailVerificationToken.user_id == stub.id,
            EmailVerificationToken.used_at.is_(None),
            EmailVerificationToken.expires_at > datetime.now(timezone.utc),
        ).limit(1)
    )
    return None if live.scalar_one_or_none() is not None else stub


async def _email_unavailable(
    session: AsyncSession, email: str, user_id: str
) -> bool:
    """Whether this address is one the flow could never actually hand over.

    Deliberately not filtered by ``deleted_at``: the unique index on
    ``users.email`` covers every row, so an address held by a soft-deleted
    account is one this flow can never hand over. Treating it as free would mean
    mailing links guaranteed to fail at the end.

    The one row that does *not* block is an abandoned signup stub — see
    :func:`_dead_signup_stub`. Confirmation clears it out of the way.
    """
    result = await session.execute(
        select(User.id).where(User.email == email, User.id != user_id).limit(1)
    )
    if result.scalar_one_or_none() is None:
        return False
    return await _dead_signup_stub(session, email, user_id) is None


@router.get("/account", response_model=AccountResponse,
            operation_id="getAccount", summary="Get the current account's identifiers")
async def get_account(
    ctx: UserContext = Depends(get_current_user),
    session: AsyncSession = Depends(get_registry_session),
):
    """The caller's own login identifiers, plus any address awaiting confirmation.

    Separate from ``GET /athlete``, which carries the training profile and no
    login identity at all — there was no endpoint that could tell the web app
    which address it is about to let you change.
    """
    result = await session.execute(
        select(User).where(User.id == ctx.user_id, User.deleted_at.is_(None))
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    pending = await _live_email_change(session, user.id, datetime.now(timezone.utc))
    return AccountResponse(
        username=user.username,
        email=user.email,
        email_verified=user.email_verified_at is not None,
        pending_email=pending.new_email if pending else None,
        # Which halves are still outstanding. The card has to name the mailbox
        # it is waiting on, and "no old side required" (a first-time set) has to
        # look different from "old side required, not done yet".
        pending_requires_old=bool(pending and pending.requires_old_confirmation),
        pending_confirmed_new=bool(pending and pending.new_confirmed_at is not None),
        pending_confirmed_old=bool(pending and pending.old_confirmed_at is not None),
    )


@router.post("/change-email", response_model=MessageResponse, status_code=202,
             operation_id="changeEmail", summary="Change the account's email address")
@limiter.limit("10/hour")
async def change_email(
    request: Request,
    body: ChangeEmailRequest,
    background: BackgroundTasks,
    ctx: UserContext = Depends(get_current_user),
    provider: EmailProvider = Depends(get_email_provider_dep),
    session: AsyncSession = Depends(get_registry_session),
):
    """Email a confirmation link to a new address; nothing moves until it is opened.

    Also the way an invite-created account *gains* an address — those have
    ``email = NULL``, which locks them out of self-serve password reset and
    login-by-email entirely. Setting one is the same operation as changing one,
    confirmed the same way.

    Requires the current password: a session on its own must not be able to
    relocate the login identifier and the password-reset target, or a borrowed
    browser becomes a permanent account takeover.

    Returns the same acknowledgement whatever happens — success, an address
    another account already holds, the caller's own current address — so the
    *content* of the answer says nothing about who has an account here
    (#102, F-06).

    Delivery is handed to a background task rather than awaited, which keeps a
    slow or wedged provider from holding the response open and closes most of
    the timing gap between the branches that send and the branches that don't.
    What remains is a database commit, not a network round trip. Note this is
    narrower than uniform: ``signup`` and ``request_password_reset`` have the
    same shape and are deliberately left alone here rather than widening a
    security fix into endpoints it wasn't otherwise touching.
    """
    if not provider.is_configured:
        # No way to confirm the new address, so no honest way to offer the
        # change. Mirrors how signup refuses when email is unavailable.
        raise HTTPException(
            status_code=404, detail="Changing your email address is not available"
        )

    result = await session.execute(
        select(User).where(User.id == ctx.user_id, User.deleted_at.is_(None))
    )
    user = result.scalar_one_or_none()
    if user is None or not await verify_password_async(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid password")

    ack = MessageResponse(detail=_CHANGE_EMAIL_ACK)
    email = str(body.new_email).lower()
    now = datetime.now(timezone.utc)

    if email == (user.email or "").lower():
        # Already theirs. Nothing to do, and saying so would confirm the address
        # to anyone who has the password but not the mailbox.
        return ack
    if await _email_unavailable(session, email, user.id):
        return ack

    old_email = user.email if user.email_verified_at is not None else None

    # One pending change at a time: the newest request wins, as it does for
    # signup verification and password reset.
    prior = await session.execute(
        select(EmailChangeToken).where(
            EmailChangeToken.user_id == user.id,
            EmailChangeToken.used_at.is_(None),
        )
    )
    for token_row in prior.scalars():
        token_row.used_at = now

    # Two independent secrets. Distinct is the point: one value satisfying both
    # sides would let whoever reads either mailbox finish alone, and the second
    # approval would be decoration.
    raw_new = secrets.token_urlsafe(32)
    raw_old = secrets.token_urlsafe(32) if old_email is not None else None
    session.add(EmailChangeToken(
        user_id=user.id,
        new_token_hash=hashlib.sha256(raw_new.encode()).hexdigest(),
        old_token_hash=(
            hashlib.sha256(raw_old.encode()).hexdigest() if raw_old else None
        ),
        new_email=email,
        # 24 hours, not the hour the one-sided flows use: this one needs two
        # mailboxes reached, and one of them is routinely a work account nobody
        # opens until morning. The window costs less than the old one did — a
        # single token no longer completes anything.
        expires_at=now + timedelta(hours=24),
    ))
    await session.commit()

    # Bind the id, not the ORM object: the task runs after the response, by
    # which time this request's session is closed and touching `user` would be
    # a detached-instance error.
    user_id = user.id

    def _url(raw: str) -> str:
        return f"{settings.frontend_url}/confirm-email-change?token={raw}"

    async def _deliver() -> None:
        try:
            await send_email_change_email(provider, to=email, action_url=_url(raw_new))
        except EmailError:
            # The request stands; the user can ask again to resend. Don't leak
            # the delivery failure into the uniform acknowledgement.
            log.exception("Failed to send email-change confirmation")

        if old_email is not None and raw_old is not None:
            # Not a notice — the authorisation. Without it, holding the password
            # alone would move the account's password-reset target, and "forgot
            # password" would then hand the whole account over permanently.
            try:
                await send_email_change_authorisation(
                    provider, to=old_email, new_email=email, action_url=_url(raw_old)
                )
            except EmailError:
                log.exception(
                    "Failed to send email-change authorisation to the old address"
                )

        try:
            await notifications.notify_user(
                user_id, notifications.EMAIL_CHANGE_REQUESTED, {"new_email": email}
            )
        except Exception:
            # Nothing is left to answer to, so a failed inbox write must not
            # take the request down after the change was already recorded.
            log.exception("Failed to write the email-change inbox message")

    background.add_task(_deliver)
    return ack


@router.post("/confirm-email-change", response_model=EmailChangeConfirmResponse,
             operation_id="confirmEmailChange", summary="Confirm a new email address")
@limiter.limit("20/hour")
async def confirm_email_change(
    request: Request,
    body: ConfirmEmailChangeRequest,
    session: AsyncSession = Depends(get_registry_session),
):
    """Stamp one side of a pending change, and apply it once both sides are in.

    Unauthenticated by design: these links are opened in whichever mailbox they
    were sent to, routinely on a different device from the one that asked. The
    token *is* the proof — holding it means holding that inbox.

    A change carries two distinct tokens, one per address, and this endpoint
    works out which it was handed by matching against both columns. Presenting
    the same one twice therefore stamps the same side twice and completes
    nothing: that is what makes the second mailbox a real requirement rather
    than a notification with a button on it.

    Sessions are left alone. Both mailboxes and the password were needed to get
    here, so there is no one to evict that isn't the owner; ``/logout-all``
    remains the control for clearing other devices.
    """
    token_hash = hashlib.sha256(body.token.encode()).hexdigest()
    result = await session.execute(
        select(EmailChangeToken).where(
            or_(
                EmailChangeToken.new_token_hash == token_hash,
                EmailChangeToken.old_token_hash == token_hash,
            )
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

    # Which half of the change did this token authorise? Re-stamping a side
    # already done is a no-op rather than an error — a double-clicked link, or a
    # mail client prefetching the URL, shouldn't read as a failure.
    if token_row.new_token_hash == token_hash:
        if token_row.new_confirmed_at is None:
            token_row.new_confirmed_at = now
    else:
        if token_row.old_confirmed_at is None:
            token_row.old_confirmed_at = now

    if not token_row.fully_confirmed:
        await session.commit()
        return EmailChangeConfirmResponse(
            complete=False,
            awaiting="old" if token_row.old_confirmed_at is None else "new",
            new_email=token_row.new_email,
        )

    # The address was free when the change was asked for; a day is long enough
    # for somebody else to have signed up with it since.
    if await _email_unavailable(session, token_row.new_email, user.id):
        raise HTTPException(
            status_code=409, detail="That email address is no longer available"
        )

    # An abandoned signup stub doesn't block the change, but it does hold the
    # address, and uq_users_email would refuse the assignment below while it
    # exists. Clearing it is what signup would have done with the same row. Such
    # a stub has no per-user database and no training data — _create_user_profile
    # runs at verification, which by definition it never reached — so the row and
    # its cascaded tokens are the whole of it.
    stub = await _dead_signup_stub(session, token_row.new_email, user.id)
    if stub is not None:
        await session.delete(stub)
        await session.flush()

    new_email = token_row.new_email
    user.email = new_email
    user.email_verified_at = now
    token_row.used_at = now
    try:
        await session.commit()
    except IntegrityError:
        # Lost a race against a concurrent signup or confirmation for the same
        # address between the check above and this commit; uq_users_email is
        # what actually decides it.
        await session.rollback()
        raise HTTPException(
            status_code=409, detail="That email address is no longer available"
        )

    await notifications.notify_user(
        user.id, notifications.EMAIL_CHANGE_CONFIRMED, {"new_email": new_email}
    )
    return EmailChangeConfirmResponse(
        complete=True, awaiting=None, new_email=new_email
    )


@router.post("/cancel-email-change", status_code=204,
             operation_id="cancelEmailChange", summary="Abandon a pending email change")
@limiter.limit("20/hour")
async def cancel_email_change(
    request: Request,
    ctx: UserContext = Depends(get_current_user),
    session: AsyncSession = Depends(get_registry_session),
):
    """Spend any outstanding change tokens without applying them.

    A mistyped address otherwise sits in the account's face for a day with no
    way to clear it, and its links stay live in whichever inboxes they reached.
    Voids both sides at once. Idempotent: cancelling nothing succeeds.
    """
    now = datetime.now(timezone.utc)
    result = await session.execute(
        select(EmailChangeToken).where(
            EmailChangeToken.user_id == ctx.user_id,
            EmailChangeToken.used_at.is_(None),
        )
    )
    for token_row in result.scalars():
        token_row.used_at = now
    await session.commit()
