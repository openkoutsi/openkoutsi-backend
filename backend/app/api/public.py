"""
Public (unauthenticated) endpoints — only for assets that browsers load directly
as image/src without an Authorization header.
"""
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.db.registry import get_registry_session
from backend.app.db.user_session import get_user_session_factory
from backend.app.models.registry_orm import InstanceSettings
from backend.app.models.user_orm import Athlete
from backend.app.services.email import get_email_provider

router = APIRouter(prefix="/public", tags=["public"])


class InstanceInfoResponse(BaseModel):
    """Non-sensitive, publicly readable instance settings."""

    admin_contact: Optional[str] = None
    privacy_policy_url: str
    # Issue #15: whether an email provider is configured (gates the self-serve
    # "email me a reset link" form) and whether self-serve signup is offered.
    email_enabled: bool = False
    allow_self_signup: bool = False


@router.get("/instance-info", response_model=InstanceInfoResponse,
            operation_id="getPublicInstanceInfo",
            summary="Get public instance info (no auth)")
async def get_instance_info(
    session: AsyncSession = Depends(get_registry_session),
) -> InstanceInfoResponse:
    """Return non-sensitive instance settings readable without authentication.

    Used by unauthenticated pages (e.g. password reset) that need the admin
    contact, and by the consent screen for the privacy-policy link. Only
    whitelisted, non-secret fields are exposed here.
    """
    result = await session.execute(select(InstanceSettings).limit(1))
    instance = result.scalar_one_or_none()
    email_enabled = get_email_provider().is_configured
    return InstanceInfoResponse(
        admin_contact=instance.admin_contact if instance else None,
        privacy_policy_url=settings.privacy_policy_url,
        email_enabled=email_enabled,
        allow_self_signup=bool(instance and instance.allow_self_signup) and email_enabled,
    )


@router.get("/users/{user_id}/avatar",
            operation_id="getPublicUserAvatar", summary="Get a user's avatar (no auth)")
async def get_avatar(user_id: str):
    """Serve a user's avatar image without requiring authentication.

    The user_id acts as the opaque reference. No sensitive data is exposed —
    only the image file itself is returned.
    """
    try:
        async with get_user_session_factory(user_id)() as session:
            result = await session.execute(
                select(Athlete).where(Athlete.global_user_id == user_id)
            )
            athlete = result.scalar_one_or_none()
    except Exception:
        raise HTTPException(status_code=404, detail="Not found")

    if athlete is None or not athlete.avatar_path:
        raise HTTPException(status_code=404, detail="No avatar set")

    path = Path(athlete.avatar_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Avatar file not found")

    return FileResponse(path)
