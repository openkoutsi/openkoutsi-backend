import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import create_access_token, hash_password
from backend.app.core.limiter import limiter
from backend.app.db.registry import get_registry_session
from backend.app.db.user_session import get_user_session_factory, init_user_db
from backend.app.models.registry_orm import User
from backend.app.models.user_orm import Athlete
from backend.app.schemas.admin import SetupRequest, SetupStatusResponse
from backend.app.schemas.auth import TokenResponse

router = APIRouter(prefix="/setup", tags=["setup"])


@router.get("/status", response_model=SetupStatusResponse,
            operation_id="getSetupStatus", summary="Whether first-run setup is needed")
async def setup_status(session: AsyncSession = Depends(get_registry_session)):
    result = await session.execute(select(func.count()).select_from(User))
    count = result.scalar_one()
    return SetupStatusResponse(needs_setup=count == 0)


@router.post("", response_model=TokenResponse, status_code=201,
             operation_id="firstRunSetup", summary="Create the first admin user")
@limiter.limit("10/hour")
async def first_run_setup(
    request: Request,
    body: SetupRequest,
    session: AsyncSession = Depends(get_registry_session),
):
    """Create the first instance administrator. Returns 409 if any user already exists."""
    existing = await session.execute(select(func.count()).select_from(User))
    if existing.scalar_one() > 0:
        raise HTTPException(status_code=409, detail="Setup already completed")

    roles = ["administrator", "user"]
    user = User(
        id=str(uuid.uuid4()),
        username=body.admin_username,
        password_hash=hash_password(body.admin_password),
        roles=json.dumps(roles),
    )
    session.add(user)
    await session.commit()

    # Create the admin's athlete profile in their own DB
    await init_user_db(user.id)
    async with get_user_session_factory(user.id)() as user_session:
        athlete = Athlete(
            id=str(uuid.uuid4()),
            global_user_id=user.id,
            name=body.admin_display_name or None,
            ftp_tests=[],
        )
        user_session.add(athlete)
        await user_session.commit()

    return TokenResponse(access_token=create_access_token(user.id, roles))
