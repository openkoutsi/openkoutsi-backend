"""
Public (unauthenticated) endpoints — only for assets that browsers load directly
as image/src without an Authorization header.
"""
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select

from backend.app.db.user_session import get_user_session_factory
from backend.app.models.user_orm import Athlete

router = APIRouter(prefix="/public", tags=["public"])


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
