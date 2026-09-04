"""Personal access tokens — issue, list and revoke (issue #46).

**Session-authenticated only.** The whole router is ``pat_forbidden``, so a token
can never create, list or revoke a token: no escalation loop, no internal minting
path. A PAT's value is that a human deliberately created it, can see it here and
can revoke it.

There is no update endpoint either: name, scopes and expiry are fixed at
creation, and widening a token means revoking it and issuing a new one. That
keeps a token id a stable answer to "what could this credential do?" rather than
one whose meaning depends on when you ask.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import UserContext, get_current_user
from backend.app.core.limiter import limiter
from backend.app.core.scopes import SCOPES, SENSITIVE_SCOPES, pat_forbidden, validate_scopes
from backend.app.db.registry import get_registry_session
from backend.app.models.registry_orm import InstanceSettings, PersonalAccessToken
from backend.app.schemas.tokens import (
    PersonalAccessTokenCreate,
    PersonalAccessTokenCreated,
    PersonalAccessTokenResponse,
    ScopeInfo,
    TokenScopesResponse,
)
from backend.app.services import personal_access_tokens as pat


router = APIRouter(prefix="/tokens", tags=["tokens"], dependencies=[pat_forbidden()])


def _to_response(token: PersonalAccessToken) -> PersonalAccessTokenResponse:
    return PersonalAccessTokenResponse(
        id=token.id,
        name=token.name,
        scopes=pat.scopes_of(token),
        status=pat.status_of(token),
        expires_at=token.expires_at,
        last_used_at=token.last_used_at,
        revoked_at=token.revoked_at,
        created_at=token.created_at,
    )


async def _require_enabled(session: AsyncSession) -> None:
    """404 when the self-hoster has switched personal access tokens off."""
    instance = (
        await session.execute(select(InstanceSettings).limit(1))
    ).scalar_one_or_none()
    if instance is not None and not instance.allow_personal_access_tokens:
        raise HTTPException(
            status_code=404, detail="Personal access tokens are not available"
        )


@router.get("/scopes", response_model=TokenScopesResponse,
            operation_id="listTokenScopes",
            summary="List the scope vocabulary a token can be granted")
async def list_scopes(
    _: UserContext = Depends(get_current_user),
    session: AsyncSession = Depends(get_registry_session),
):
    await _require_enabled(session)
    return TokenScopesResponse(
        scopes=[
            ScopeInfo(
                name=name,
                description=description,
                sensitive=name in SENSITIVE_SCOPES,
            )
            for name, description in SCOPES.items()
        ],
        allowed_lifetime_days=list(pat.ALLOWED_LIFETIME_DAYS),
    )


@router.post("", response_model=PersonalAccessTokenCreated, status_code=201,
             operation_id="createPersonalAccessToken",
             summary="Create a personal access token")
@limiter.limit("20/hour")
async def create_token(
    request: Request,
    body: PersonalAccessTokenCreate,
    ctx: UserContext = Depends(get_current_user),
    session: AsyncSession = Depends(get_registry_session),
):
    """Mint a token and return its secret — the only time it is ever returned."""
    await _require_enabled(session)

    try:
        scopes = validate_scopes(body.scopes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not scopes:
        raise HTTPException(status_code=400, detail="At least one scope is required.")

    try:
        expires_at = pat.expires_at_for(body.expires_in_days)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    token_id, raw_token, token_hash = pat.mint_token()
    token = PersonalAccessToken(
        id=token_id,
        user_id=ctx.user_id,
        token_hash=token_hash,
        name=body.name.strip(),
        scopes=json.dumps(scopes),
        expires_at=expires_at,
    )
    session.add(token)
    await session.commit()
    await session.refresh(token)

    return PersonalAccessTokenCreated(
        **_to_response(token).model_dump(), token=raw_token
    )


@router.get("", response_model=list[PersonalAccessTokenResponse],
            operation_id="listPersonalAccessTokens",
            summary="List your personal access tokens")
async def list_tokens(
    ctx: UserContext = Depends(get_current_user),
    session: AsyncSession = Depends(get_registry_session),
):
    """Every token the caller holds, live and dead.

    Expired and revoked rows are retained rather than pruned, so the list is the
    user's own record of what they issued — the UI groups active apart from
    expired/revoked the way the invitations tab splits pending from used.
    """
    await _require_enabled(session)
    result = await session.execute(
        select(PersonalAccessToken)
        .where(PersonalAccessToken.user_id == ctx.user_id)
        .order_by(PersonalAccessToken.created_at.desc())
    )
    return [_to_response(token) for token in result.scalars().all()]


@router.delete("/{token_id}", status_code=204,
               operation_id="revokePersonalAccessToken",
               summary="Revoke a personal access token")
async def revoke_token(
    token_id: str,
    ctx: UserContext = Depends(get_current_user),
    session: AsyncSession = Depends(get_registry_session),
):
    """Withdraw a token immediately — no cache, no grace window.

    The row survives, hash included: a later attempt with it is then logged as a
    revoked-token attempt rather than an unknown one, which is a different event
    and worth being able to tell apart.
    """
    result = await session.execute(
        select(PersonalAccessToken).where(
            PersonalAccessToken.id == token_id,
            PersonalAccessToken.user_id == ctx.user_id,
        )
    )
    token = result.scalar_one_or_none()
    if token is None:
        raise HTTPException(status_code=404, detail="Token not found")
    if token.revoked_at is None:
        token.revoked_at = datetime.now(timezone.utc)
        await session.commit()
