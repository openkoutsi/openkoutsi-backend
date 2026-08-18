import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import DateTime, exists, func, insert, literal, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import create_access_token, hash_password_async
from backend.app.core.limiter import limiter
from backend.app.core.scopes import pat_forbidden
from backend.app.db.registry import get_registry_session
from backend.app.db.user_session import get_user_session_factory, init_user_db
from backend.app.models.registry_orm import User
from backend.app.models.user_orm import Athlete
from backend.app.schemas.admin import SetupRequest, SetupStatusResponse
from backend.app.schemas.auth import TokenResponse


router = APIRouter(prefix="/setup", tags=["setup"], dependencies=[pat_forbidden()])


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
    # Cheap path for the ordinary case — an instance that is already set up
    # answers without paying for a bcrypt hash. Not the guard: the statement
    # below is what actually decides.
    existing = await session.execute(select(func.count()).select_from(User))
    if existing.scalar_one() > 0:
        raise HTTPException(status_code=409, detail="Setup already completed")

    roles = ["administrator", "user"]
    user_id = str(uuid.uuid4())
    password_hash = await hash_password_async(body.admin_password)

    # Check and insert in one statement. Counting first and inserting after
    # leaves an `await` between the two — and the hash above is a quarter of a
    # second of it — so two requests arriving before either commits both saw
    # zero users and both created an administrator (issue #102, F-13). This
    # endpoint is unauthenticated by definition, and the window is open on a
    # freshly deployed instance, which is exactly when an admin is about to
    # make this request themselves.
    #
    # `INSERT ... SELECT ... WHERE NOT EXISTS` moves the check inside the
    # write, where the database settles it: the second writer inserts nothing
    # and says so in its row count. A lease would also serialise this, but a
    # statement the database cannot interleave needs no TTL to be right.
    claimed = await session.execute(
        insert(User).from_select(
            ["id", "username", "password_hash", "roles", "created_at", "token_version"],
            select(
                literal(user_id),
                literal(body.admin_username),
                literal(password_hash),
                literal(json.dumps(roles)),
                literal(datetime.now(timezone.utc), DateTime(timezone=True)),
                literal(0),
            ).where(~exists(select(User.id))),
        )
    )
    await session.commit()

    if claimed.rowcount == 0:
        # Someone else got there between our count and our insert.
        raise HTTPException(status_code=409, detail="Setup already completed")

    # Create the admin's athlete profile in their own DB
    await init_user_db(user_id)
    async with get_user_session_factory(user_id)() as user_session:
        athlete = Athlete(
            id=str(uuid.uuid4()),
            global_user_id=user_id,
            name=body.admin_display_name or None,
            ftp_tests=[],
        )
        user_session.add(athlete)
        await user_session.commit()

    return TokenResponse(
        access_token=create_access_token(user_id, roles, token_version=0)
    )
