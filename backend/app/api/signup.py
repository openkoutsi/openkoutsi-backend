import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import hash_password
from backend.app.core.limiter import limiter
from backend.app.db.registry import get_registry_session
from backend.app.db.team_session import init_team_db, get_team_session_factory
from backend.app.models.registry_orm import Team, TeamMembership, User
from backend.app.schemas.teams import TeamSignupRequest, TeamSignupResponse
from backend.app.services import notifications

router = APIRouter(prefix="/teams", tags=["signup"])


@router.post("", response_model=TeamSignupResponse, status_code=201)
@limiter.limit("5/hour")
async def create_team(
    request: Request,
    body: TeamSignupRequest,
    session: AsyncSession = Depends(get_registry_session),
):
    """Self-serve team creation. New teams start in 'pending' status and require
    superadmin approval before members can log in."""
    slug_check = await session.execute(select(Team).where(Team.slug == body.slug))
    if slug_check.scalar_one_or_none() is not None:
        raise HTTPException(status_code=400, detail="Slug already taken")

    username_check = await session.execute(
        select(User).where(User.username == body.admin_username)
    )
    if username_check.scalar_one_or_none() is not None:
        raise HTTPException(status_code=400, detail="Username already taken")

    team = Team(
        id=str(uuid.uuid4()),
        slug=body.slug,
        name=body.team_name,
        status="pending",
    )
    session.add(team)
    await session.flush()

    user = User(
        id=str(uuid.uuid4()),
        username=body.admin_username,
        password_hash=hash_password(body.admin_password),
    )
    session.add(user)
    await session.flush()

    membership = TeamMembership(
        team_id=team.id,
        user_id=user.id,
        roles=json.dumps(["administrator", "user"]),
    )
    session.add(membership)
    await session.commit()

    await init_team_db(team.id)
    from backend.app.models.team_orm import Athlete
    async with get_team_session_factory(team.id)() as team_session:
        athlete = Athlete(
            id=str(uuid.uuid4()),
            global_user_id=user.id,
            name=body.admin_display_name or None,
            ftp_tests=[],
        )
        team_session.add(athlete)
        await team_session.commit()

    await notifications.notify_superadmin(
        notifications.TEAM_REQUEST,
        {
            "team_id": team.id,
            "team_name": team.name,
            "team_slug": team.slug,
            "admin_username": user.username,
        },
    )

    return TeamSignupResponse(
        id=team.id,
        slug=team.slug,
        name=team.name,
        status=team.status,
        created_at=team.created_at,
    )
