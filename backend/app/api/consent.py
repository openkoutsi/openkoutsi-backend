from datetime import datetime, timezone

CURRENT_CONSENT_VERSION = "1.0"

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import TeamContext, get_current_user
from backend.app.db.registry import get_registry_session
from backend.app.models.registry_orm import DataConsent, Team
from backend.app.schemas.teams import ConsentRequest, ConsentResponse

router = APIRouter(prefix="/teams/{slug}", tags=["consent"])


@router.post("/consent", response_model=ConsentResponse, status_code=200)
async def record_consent(
    slug: str,
    body: ConsentRequest,
    ctx: TeamContext = Depends(get_current_user),
    session: AsyncSession = Depends(get_registry_session),
):
    result = await session.execute(select(Team).where(Team.slug == slug))
    team = result.scalar_one_or_none()
    if team is None:
        raise HTTPException(status_code=404, detail="Team not found")

    if ctx.team_id != team.id:
        raise HTTPException(status_code=403, detail="Not a member of this team")

    existing = await session.execute(
        select(DataConsent).where(
            DataConsent.user_id == ctx.user_id,
            DataConsent.team_id == team.id,
        )
    )
    consent = existing.scalar_one_or_none()

    now = datetime.now(timezone.utc)
    if consent is None:
        consent = DataConsent(
            user_id=ctx.user_id,
            team_id=team.id,
            consented_at=now,
            consent_version=body.consent_version or CURRENT_CONSENT_VERSION,
        )
        session.add(consent)
    else:
        consent.consented_at = now
        consent.consent_version = body.consent_version or CURRENT_CONSENT_VERSION

    await session.commit()
    await session.refresh(consent)
    return ConsentResponse(consented_at=consent.consented_at, consent_version=consent.consent_version)
