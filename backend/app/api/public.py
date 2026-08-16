"""
Public (unauthenticated) endpoints — only for assets that browsers load directly
as image/src without an Authorization header.
"""
import uuid
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
from backend.app.models.registry_orm import InstanceSettings, User
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
    # Issue #46: whether users may issue personal access tokens on this
    # instance. Exposed here (rather than only on the admin settings endpoint)
    # so the settings card can hide itself without an admin round trip.
    allow_personal_access_tokens: bool = True


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
        allow_personal_access_tokens=(
            bool(instance.allow_personal_access_tokens) if instance else True
        ),
    )


def _is_canonical_uuid(user_id: str) -> bool:
    """True when *user_id* is a UUID in the exact form the registry stores.

    The path segment names a directory (``<data_dir>/users/<user_id>/``), so it
    is checked before anything opens a database with it. Canonical form only:
    every real id is ``str(uuid.uuid4())``, and accepting the ``{...}``,
    ``urn:uuid:`` or undashed spellings would let one account's id name several
    different directories.
    """
    try:
        return str(uuid.UUID(user_id)) == user_id
    except (AttributeError, TypeError, ValueError):
        return False


@router.get("/users/{user_id}/avatar",
            operation_id="getPublicUserAvatar", summary="Get a user's avatar (no auth)")
async def get_avatar(
    user_id: str,
    registry_session: AsyncSession = Depends(get_registry_session),
):
    """Serve a user's avatar image without requiring authentication.

    The user_id acts as the opaque reference. No sensitive data is exposed —
    only the image file itself is returned. An id that is not a known user's
    is refused before anything else happens.
    """
    # The id is resolved against the registry *before* a per-user session is
    # opened with it. This route takes no credential and no rate limit, and
    # opening a session used to create the user's directory and database as a
    # side effect, so an anonymous caller got a new directory and three files
    # per request and evicted every real user's cached engine on the way past
    # (issue #102, F-03). Engine construction no longer creates anything (see
    # db/user_session.py), and this keeps an unknown id from reaching it at all.
    #
    # Kept out of the docstring deliberately: FastAPI publishes that verbatim
    # as this endpoint's public OpenAPI description.
    if not _is_canonical_uuid(user_id):
        raise HTTPException(status_code=404, detail="Not found")

    known_user = await registry_session.execute(
        select(User.id).where(User.id == user_id, User.deleted_at.is_(None))
    )
    if known_user.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Not found")

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
