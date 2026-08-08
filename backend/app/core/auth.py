from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import bcrypt as _bcrypt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core import audit
from backend.app.core.config import settings
from backend.app.core.scopes import PatAccess, access_for_request
from backend.app.db.registry import get_registry_session

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

#: ``UserContext.token_kind`` values.
SESSION_TOKEN = "session"
PERSONAL_ACCESS_TOKEN = "pat"


@dataclass
class UserContext:
    """Identity extracted from a validated credential.

    The instance is single-tenant: there is no team. Every user's data lives in
    their own per-user DB, addressed by ``user_id``.

    ``scopes`` is ``None`` for a session token, meaning full access — so widening
    this dataclass for personal access tokens (issue #46) changed no existing
    route's behaviour. A PAT carries an explicit list instead, and reaches only
    the routes that declared themselves reachable (``core.scopes``).
    """
    user_id: str    # global user UUID (registry users.id)
    roles: list[str]
    # None ⇒ a session token with full access; a list ⇒ a scoped credential.
    scopes: list[str] | None = None
    token_kind: str = SESSION_TOKEN
    # Registry ``personal_access_tokens.id`` when this is a PAT — the stable
    # principal the audit log and the rate limiter key on.
    token_id: str | None = None

    @property
    def is_pat(self) -> bool:
        return self.token_kind == PERSONAL_ACCESS_TOKEN

    @property
    def is_admin(self) -> bool:
        # A personal access token is never administrative, whatever its owner's
        # roles say. Admin status must not widen the athlete-data surface, and
        # the /api/admin router is closed to tokens outright; this keeps the
        # answer the same for anything that asks the context directly.
        if self.is_pat:
            return False
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


def _credentials_exception() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _roles_of(user) -> list[str]:
    try:
        return json.loads(user.roles) if user.roles else []
    except (TypeError, ValueError):
        return []


async def validate_personal_access_token(
    raw_token: str,
    registry_session: AsyncSession,
    *,
    method: str,
    path: str,
):
    """Everything about a ``okp_`` token *except* what it is allowed to do.

    Is this credential well-formed, known, unrevoked, unexpired, enabled on this
    instance, and whose is it? Returns ``(UserContext, token_row)``; every
    failure raises the same opaque 401 and is written to the audit log first.

    Split out from :func:`_resolve_personal_access_token` because there is one
    other surface that authorizes differently: the MCP server (issue #42) gates
    on the *tool* named in the request body rather than on the route, since no
    single declaration on ``POST /mcp`` could be right for nine differently-scoped
    tools. It therefore never reaches the route-policy map at all. Splitting the
    policy question off lets it reuse this without becoming a second identity
    path — the part that decides *who the caller is*, and therefore which
    database and encryption key their request reaches, stays a single
    implementation.
    """
    from backend.app.models.registry_orm import InstanceSettings, User
    from backend.app.services import personal_access_tokens as pat

    def deny(outcome: str, token_id: str | None, **fields) -> HTTPException:
        audit.pat_request(
            outcome=outcome, token_id=token_id, method=method, path=path, **fields
        )
        return _credentials_exception()

    parsed = pat.parse_token(raw_token)
    if parsed is None:
        raise deny(audit.UNKNOWN_TOKEN, None)
    token_id, secret = parsed

    # The instance kill switch refuses *authentication*, not just issuance. If it
    # only blocked the create endpoint, every token handed out beforehand would
    # keep working and the admin would have been told a comforting untruth.
    instance = (
        await registry_session.execute(select(InstanceSettings).limit(1))
    ).scalar_one_or_none()
    if instance is not None and not instance.allow_personal_access_tokens:
        raise deny(audit.DISABLED, token_id)

    token = await pat.load_by_id(registry_session, token_id)
    if token is None:
        raise deny(audit.UNKNOWN_TOKEN, token_id)
    if not pat.verify_secret(secret, token.token_hash):
        raise deny(audit.BAD_SECRET, token_id, user_id=token.user_id)

    status_now = pat.status_of(token)
    if status_now == "revoked":
        raise deny(audit.REVOKED, token_id, user_id=token.user_id)
    if status_now == "expired":
        raise deny(audit.EXPIRED, token_id, user_id=token.user_id)

    user = (
        await registry_session.execute(
            select(User).where(User.id == token.user_id, User.deleted_at.is_(None))
        )
    ).scalar_one_or_none()
    if user is None:
        raise deny(audit.UNKNOWN_TOKEN, token_id)

    return (
        UserContext(
            user_id=user.id,
            roles=_roles_of(user),
            scopes=pat.scopes_of(token),
            token_kind=PERSONAL_ACCESS_TOKEN,
            token_id=token_id,
        ),
        token,
    )


async def _resolve_personal_access_token(
    request: Request,
    raw_token: str,
    registry_session: AsyncSession,
) -> UserContext:
    """Authenticate a ``okp_``-prefixed personal access token (issue #46).

    Deliberately inside ``get_current_user`` rather than beside it: a second
    identity path would be a second place for the per-user isolation guarantee to
    be lost. Everything downstream — ``get_ctx_and_session``,
    ``get_ctx_session_athlete``, ``require_consent``, every
    ``Activity.athlete_id == athlete.id`` filter — sees an ordinary
    ``UserContext`` and works unchanged.
    """
    from backend.app.services import personal_access_tokens as pat

    method = request.method
    path = request.url.path

    ctx, token = await validate_personal_access_token(
        raw_token, registry_session, method=method, path=path
    )
    token_id = ctx.token_id
    scopes = ctx.scopes or []

    # Default-deny: a route that declared nothing is absent from the map and so
    # unreachable, and the hole cannot arrive silently with a router added later.
    # Resolved from the route rather than from what has run so far — see
    # `core.scopes` for why that distinction is load-bearing.
    access: PatAccess | None = access_for_request(request)
    required = access.scope_for(method) if access is not None else None
    if access is None or not access.allowed or required is None:
        audit.pat_request(
            outcome=audit.DENIED_ROUTE,
            token_id=token_id,
            user_id=token.user_id,
            method=method,
            path=path,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint is not available to personal access tokens.",
        )
    if required not in scopes:
        audit.pat_request(
            outcome=audit.DENIED_SCOPE,
            token_id=token_id,
            user_id=token.user_id,
            method=method,
            path=path,
            scope=required,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"This token is missing the '{required}' scope.",
        )

    await pat.touch_last_used(registry_session, token)
    # The rate limiter keys on the *user* rather than the address: one script
    # hammering from one IP is not one anonymous visitor. Not on the token —
    # tokens can be minted freely, so per-token buckets would make the limit
    # multiplicative in a number nothing caps. The token id is recorded too, for
    # anything that wants per-token attribution rather than throttling.
    request.state.pat_user_id = token.user_id
    request.state.pat_token_id = token_id
    audit.pat_request(
        outcome=audit.OK,
        token_id=token_id,
        user_id=token.user_id,
        method=method,
        path=path,
        scope=required,
    )

    await registry_session.close()
    return ctx


async def validate_session_token(
    raw_token: str, registry_session: AsyncSession
) -> UserContext:
    """Identity from a session JWT, with the registry row as the authority.

    Roles are re-read from the registry rather than trusted from the token
    claim, so a role change takes effect without waiting for token expiry.
    """
    from backend.app.models.registry_orm import User

    credentials_exception = _credentials_exception()
    try:
        payload = decode_token(raw_token)
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
    return UserContext(user_id=user_id, roles=_roles_of(user))


async def authenticate_bearer(
    raw_token: str,
    registry_session: AsyncSession,
    *,
    method: str,
    path: str,
) -> UserContext:
    """Identity from a bearer value, with **no route policy applied**.

    For surfaces that authorize per *operation* rather than per route: the MCP
    server (issue #42) is mounted as a sub-application, so ``build_access_map``
    never saw it, and the scope a call needs is a property of the tool being
    invoked rather than of the URL it arrived at. Such a surface must therefore
    do its own default-deny check — for the tool layer that is
    ``backend.app.mcp.registry``, where every tool declares its scopes and an
    undeclared one cannot be registered at all.

    A personal access token still has to be live, unrevoked and enabled here;
    only the *authorization* question is deferred.
    """
    from backend.app.services import personal_access_tokens as pat

    if pat.looks_like_pat(raw_token):
        ctx, token = await validate_personal_access_token(
            raw_token, registry_session, method=method, path=path
        )
        await pat.touch_last_used(registry_session, token)
        return ctx
    return await validate_session_token(raw_token, registry_session)


async def get_current_user(
    request: Request,
    token: str = Depends(oauth2_scheme),
    registry_session: AsyncSession = Depends(get_registry_session),
) -> UserContext:
    from backend.app.services.personal_access_tokens import looks_like_pat

    # Route on the prefix *before* attempting a JWT decode: a PAT is opaque and
    # DB-backed, so it would only ever fail to decode.
    if looks_like_pat(token):
        return await _resolve_personal_access_token(request, token, registry_session)

    ctx = await validate_session_token(token, registry_session)
    # Release the pool connection immediately — the user object is no longer
    # needed, but the dependency would otherwise keep the session (and its pool
    # slot) alive until request end while the per-user session is in use.
    await registry_session.close()
    return ctx
