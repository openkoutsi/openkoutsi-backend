"""
Public (unauthenticated) endpoints — only for assets that browsers load directly
as image/src without an Authorization header.
"""
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.app.db.team_session import get_team_session_factory
from backend.app.models.team_orm import Athlete
from sqlalchemy import select

router = APIRouter(prefix="/public", tags=["public"])


@router.get("/teams/{team_id}/avatar/{athlete_id}")
async def get_avatar(team_id: str, athlete_id: str):
    """Serve an athlete's avatar image without requiring authentication.

    The team_id + athlete_id pair acts as the opaque reference. No sensitive
    data is exposed — only the image file itself is returned.
    """
    try:
        async with get_team_session_factory(team_id)() as session:
            result = await session.execute(
                select(Athlete).where(Athlete.id == athlete_id)
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
