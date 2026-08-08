from datetime import datetime, timezone

CURRENT_CONSENT_VERSION = "1.0"

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import UserContext, get_current_user
from backend.app.core.scopes import pat_forbidden
from backend.app.db.registry import get_registry_session
from backend.app.models.registry_orm import User
from backend.app.schemas.admin import ConsentRequest, ConsentResponse


router = APIRouter(prefix="/consent", tags=["consent"], dependencies=[pat_forbidden()])


async def has_consent(user_id: str, session: AsyncSession) -> bool:
    """Whether ``user_id`` has accepted the *current* privacy policy version.

    The plain-function half of :func:`require_consent`, for callers that are not
    a route: the MCP tool layer (issue #42) applies the same gate per tool
    invocation, and a second copy of "what counts as consent" is exactly the
    kind of thing that drifts once the policy version moves.
    """
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    return (
        user is not None
        and user.consented_at is not None
        and user.consent_version == CURRENT_CONSENT_VERSION
    )


async def require_consent(
    ctx: UserContext = Depends(get_current_user),
    session: AsyncSession = Depends(get_registry_session),
) -> None:
    """Block a route until the user has accepted the current privacy policy.

    Applied as a route dependency on data-ingestion entry points (provider
    connect, manual upload) so the API — not just the UI redirect — refuses to
    start processing health data before consent is on record. Returns 403 when
    consent is missing or predates the current policy version.
    """
    if not await has_consent(ctx.user_id, session):
        raise HTTPException(
            status_code=403,
            detail="Consent to data processing is required before this action.",
        )


@router.post("", response_model=ConsentResponse, status_code=200,
             operation_id="recordConsent", summary="Record data-processing consent")
async def record_consent(
    body: ConsentRequest,
    ctx: UserContext = Depends(get_current_user),
    session: AsyncSession = Depends(get_registry_session),
):
    result = await session.execute(select(User).where(User.id == ctx.user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    now = datetime.now(timezone.utc)
    user.consented_at = now
    user.consent_version = body.consent_version or CURRENT_CONSENT_VERSION
    await session.commit()
    await session.refresh(user)
    return ConsentResponse(consented_at=user.consented_at, consent_version=user.consent_version)
