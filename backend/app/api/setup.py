import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import create_access_token, create_refresh_token, hash_password
from backend.app.core.limiter import limiter
from backend.app.db.registry import get_registry_session
from backend.app.db.team_session import init_team_db, get_team_session_factory
from backend.app.models.registry_orm import Team, TeamMembership, User
from backend.app.schemas.teams import SetupRequest, SetupStatusResponse, TeamResponse
from backend.app.schemas.auth import TokenResponse

router = APIRouter(prefix="/setup", tags=["setup"])


@router.get("/status", response_model=SetupStatusResponse)
async def setup_status(session: AsyncSession = Depends(get_registry_session)):
    result = await session.execute(select(func.count()).select_from(Team))
    count = result.scalar_one()
    return SetupStatusResponse(needs_setup=count == 0)


@router.post("", response_model=TokenResponse, status_code=201)
@limiter.limit("10/hour")
async def first_run_setup(
    request: Request,
    body: SetupRequest,
    session: AsyncSession = Depends(get_registry_session),
):
    """Create the first team and admin user. Returns 409 if any team already exists."""
    existing = await session.execute(select(func.count()).select_from(Team))
    if existing.scalar_one() > 0:
        raise HTTPException(status_code=409, detail="Setup already completed")

    slug_check = await session.execute(select(Team).where(Team.slug == body.slug))
    if slug_check.scalar_one_or_none() is not None:
        raise HTTPException(status_code=400, detail="Slug already taken")

    team = Team(id=str(uuid.uuid4()), slug=body.slug, name=body.team_name, status="active")
    session.add(team)
    await session.flush()

    user = User(
        id=str(uuid.uuid4()),
        username=body.admin_username,
        password_hash=hash_password(body.admin_password),
    )
    session.add(user)
    await session.flush()

    roles = ["administrator", "user"]
    membership = TeamMembership(
        team_id=team.id,
        user_id=user.id,
        roles=json.dumps(roles),
    )
    session.add(membership)
    await session.commit()

    # Create the team DB and the admin's athlete profile
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

    access_token = create_access_token(user.id, team.id, roles)
    return TokenResponse(access_token=access_token)
