from datetime import datetime, timezone

CURRENT_CONSENT_VERSION = "1.0"

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import UserContext, get_current_user
from backend.app.db.registry import get_registry_session
from backend.app.models.registry_orm import User
from backend.app.schemas.admin import ConsentRequest, ConsentResponse

router = APIRouter(prefix="/consent", tags=["consent"])


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
