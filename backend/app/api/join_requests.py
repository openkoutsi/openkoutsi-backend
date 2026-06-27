"""Self-serve "request to join a team" flow.

Submission is public (rate-limited); listing and approve/reject are admin-only.
Approving a request creates the user account, membership and athlete profile,
mirroring the invite-based registration path in ``api/auth.py``.
"""
import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.teams import _require_admin, _resolve_team
from backend.app.core.auth import TeamContext, get_current_user, hash_password
from backend.app.core.limiter import limiter
from backend.app.db.registry import get_registry_session
from backend.app.db.team_session import get_team_session_factory, init_team_db
from backend.app.models.registry_orm import JoinRequest, Team, TeamMembership, User
from backend.app.models.team_orm import Athlete
from backend.app.schemas.join_requests import JoinRequestCreate, JoinRequestResponse
from backend.app.services import notifications

router = APIRouter(prefix="/teams/{slug}", tags=["join-requests"])


def _to_response(jr: JoinRequest, team_slug: str) -> JoinRequestResponse:
    return JoinRequestResponse(
        id=jr.id,
        team_slug=team_slug,
        username=jr.username,
        display_name=jr.display_name,
        message=jr.message,
        status=jr.status,
        created_at=jr.created_at,
    )


@router.post("/join-requests", response_model=JoinRequestResponse, status_code=201)
@limiter.limit("5/hour")
async def create_join_request(
    request: Request,
    slug: str,
    body: JoinRequestCreate,
    session: AsyncSession = Depends(get_registry_session),
):
    team = await _resolve_team(slug, session)
    if team.status != "active":
        raise HTTPException(status_code=403, detail="Team is not accepting join requests")

    # Reject any username that already exists globally. This is a public,
    # unauthenticated flow: we have no proof the requester controls an existing
    # account, so approving such a request must never attach a stranger to it.
    # Existing users who want to join another team should use an invite.
    existing_user = await session.execute(
        select(User).where(User.username == body.username)
    )
    user = existing_user.scalar_one_or_none()
    if user is not None and user.deleted_at is None:
        raise HTTPException(status_code=400, detail="Username not available")

    # One pending request per username per team.
    dup = await session.execute(
        select(JoinRequest).where(
            JoinRequest.team_id == team.id,
            JoinRequest.username == body.username,
            JoinRequest.status == "pending",
        )
    )
    if dup.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=400, detail="A pending request already exists for this username"
        )

    jr = JoinRequest(
        id=str(uuid.uuid4()),
        team_id=team.id,
        username=body.username,
        password_hash=hash_password(body.password),
        display_name=body.display_name or None,
        message=body.message or None,
    )
    session.add(jr)
    await session.commit()

    await notifications.notify_team_admins(
        session,
        team.id,
        notifications.JOIN_REQUEST,
        {
            "join_request_id": jr.id,
            "username": jr.username,
            "display_name": jr.display_name,
            "message": jr.message,
            "team_name": team.name,
            "team_slug": team.slug,
        },
    )
    return _to_response(jr, team.slug)


@router.get("/join-requests", response_model=list[JoinRequestResponse])
async def list_join_requests(
    slug: str,
    status: str = Query(default="pending"),
    ctx: TeamContext = Depends(get_current_user),
    session: AsyncSession = Depends(get_registry_session),
):
    team = await _resolve_team(slug, session)
    _require_admin(ctx, team)

    query = select(JoinRequest).where(JoinRequest.team_id == team.id)
    if status:
        query = query.where(JoinRequest.status == status)
    result = await session.execute(query.order_by(JoinRequest.created_at.desc()))
    return [_to_response(jr, team.slug) for jr in result.scalars().all()]


async def _get_pending(
    request_id: str, team: Team, session: AsyncSession
) -> JoinRequest:
    result = await session.execute(
        select(JoinRequest).where(
            JoinRequest.id == request_id, JoinRequest.team_id == team.id
        )
    )
    jr = result.scalar_one_or_none()
    if jr is None:
        raise HTTPException(status_code=404, detail="Join request not found")
    if jr.status != "pending":
        raise HTTPException(status_code=400, detail="Join request already decided")
    return jr


@router.post("/join-requests/{request_id}/approve", response_model=JoinRequestResponse)
async def approve_join_request(
    slug: str,
    request_id: str,
    ctx: TeamContext = Depends(get_current_user),
    session: AsyncSession = Depends(get_registry_session),
):
    team = await _resolve_team(slug, session)
    _require_admin(ctx, team)
    jr = await _get_pending(request_id, team, session)

    # The username was guaranteed not to exist when the request was submitted.
    # If it exists now it was claimed by someone else in the meantime, and we
    # cannot prove the requester owns it — refuse rather than attach silently.
    existing = await session.execute(select(User).where(User.username == jr.username))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status_code=400, detail="Username is no longer available")

    user = User(
        id=str(uuid.uuid4()),
        username=jr.username,
        password_hash=jr.password_hash,
    )
    session.add(user)
    await session.flush()

    session.add(
        TeamMembership(
            team_id=team.id,
            user_id=user.id,
            roles=json.dumps(["user"]),
        )
    )

    jr.status = "approved"
    jr.decided_at = datetime.now(timezone.utc)
    jr.decided_by_user_id = ctx.user_id
    await session.commit()

    await init_team_db(team.id)
    async with get_team_session_factory(team.id)() as team_session:
        team_session.add(
            Athlete(
                id=str(uuid.uuid4()),
                global_user_id=user.id,
                name=jr.display_name or None,
                ftp_tests=[],
            )
        )
        await team_session.commit()

    return _to_response(jr, team.slug)


@router.post("/join-requests/{request_id}/reject", response_model=JoinRequestResponse)
async def reject_join_request(
    slug: str,
    request_id: str,
    ctx: TeamContext = Depends(get_current_user),
    session: AsyncSession = Depends(get_registry_session),
):
    team = await _resolve_team(slug, session)
    _require_admin(ctx, team)
    jr = await _get_pending(request_id, team, session)

    jr.status = "rejected"
    jr.decided_at = datetime.now(timezone.utc)
    jr.decided_by_user_id = ctx.user_id
    await session.commit()
    return _to_response(jr, team.slug)
