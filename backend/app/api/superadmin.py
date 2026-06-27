from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

import json

from backend.app.core.config import settings
from backend.app.db.registry import get_registry_session
from backend.app.db.user_session import get_user_session_factory, init_user_db
from backend.app.models.message_orm import Message
from backend.app.models.registry_orm import DataConsent, Team, TeamMembership, User
from backend.app.schemas.messages import MessageResponse, UnreadCountResponse
from backend.app.schemas.teams import SuperadminTeamResponse, SuperadminUserResponse, SuperadminUserTeam
from backend.app.services.notifications import SUPERADMIN_MAILBOX

router = APIRouter(prefix="/superadmin", tags=["superadmin"])


def _require_secret(x_superadmin_secret: str | None = Header(default=None)) -> None:
    if not settings.superadmin_secret:
        raise HTTPException(status_code=503, detail="Superadmin not configured")
    if x_superadmin_secret != settings.superadmin_secret:
        raise HTTPException(status_code=403, detail="Invalid superadmin secret")


@router.get("/teams", response_model=list[SuperadminTeamResponse])
async def list_teams(
    _: None = Depends(_require_secret),
    session: AsyncSession = Depends(get_registry_session),
):
    result = await session.execute(select(Team).order_by(Team.created_at.desc()))
    teams = result.scalars().all()

    counts_result = await session.execute(
        select(TeamMembership.team_id, func.count().label("n"))
        .group_by(TeamMembership.team_id)
    )
    counts = {row.team_id: row.n for row in counts_result}

    consent_counts_result = await session.execute(
        select(DataConsent.team_id, func.count().label("n"))
        .group_by(DataConsent.team_id)
    )
    consent_counts = {row.team_id: row.n for row in consent_counts_result}

    return [
        SuperadminTeamResponse(
            id=t.id,
            slug=t.slug,
            name=t.name,
            status=t.status,
            created_at=t.created_at,
            member_count=counts.get(t.id, 0),
            consented_count=consent_counts.get(t.id, 0),
        )
        for t in teams
    ]


@router.get("/users", response_model=list[SuperadminUserResponse])
async def list_users(
    _: None = Depends(_require_secret),
    session: AsyncSession = Depends(get_registry_session),
):
    users_result = await session.execute(
        select(User).where(User.deleted_at.is_(None)).order_by(User.created_at.desc())
    )
    users = users_result.scalars().all()

    memberships_result = await session.execute(
        select(TeamMembership, Team)
        .join(Team, TeamMembership.team_id == Team.id)
    )
    memberships_by_user: dict[str, list[tuple]] = {}
    for membership, team in memberships_result:
        memberships_by_user.setdefault(membership.user_id, []).append((membership, team))

    consents_result = await session.execute(select(DataConsent))
    consents: dict[tuple[str, str], DataConsent] = {
        (c.user_id, c.team_id): c for c in consents_result.scalars().all()
    }

    response = []
    for user in users:
        teams = []
        for membership, team in memberships_by_user.get(user.id, []):
            consent = consents.get((user.id, team.id))
            teams.append(SuperadminUserTeam(
                team_id=team.id,
                team_slug=team.slug,
                team_name=team.name,
                roles=json.loads(membership.roles),
                joined_at=membership.joined_at,
                consented_at=consent.consented_at if consent else None,
                consent_version=consent.consent_version if consent else None,
            ))
        response.append(SuperadminUserResponse(
            id=user.id,
            username=user.username,
            created_at=user.created_at,
            teams=teams,
        ))
    return response


@router.post("/teams/{team_id}/approve", response_model=SuperadminTeamResponse)
async def approve_team(
    team_id: str,
    _: None = Depends(_require_secret),
    session: AsyncSession = Depends(get_registry_session),
):
    result = await session.execute(select(Team).where(Team.id == team_id))
    team = result.scalar_one_or_none()
    if team is None:
        raise HTTPException(status_code=404, detail="Team not found")

    team.status = "active"
    await session.commit()

    count_result = await session.execute(
        select(func.count()).select_from(TeamMembership).where(TeamMembership.team_id == team_id)
    )
    return SuperadminTeamResponse(
        id=team.id,
        slug=team.slug,
        name=team.name,
        status=team.status,
        created_at=team.created_at,
        member_count=count_result.scalar_one(),
    )


@router.delete("/teams/{team_id}", status_code=204)
async def delete_team(
    team_id: str,
    _: None = Depends(_require_secret),
    session: AsyncSession = Depends(get_registry_session),
):
    result = await session.execute(select(Team).where(Team.id == team_id))
    team = result.scalar_one_or_none()
    if team is None:
        raise HTTPException(status_code=404, detail="Team not found")

    await session.delete(team)
    await session.commit()


# ── Superadmin inbox ──────────────────────────────────────────────────────────
# Messages addressed to the superadmin (who has no user account) live in a
# reserved per-user mailbox keyed by SUPERADMIN_MAILBOX.


async def _get_superadmin_message(message_id: str, session: AsyncSession) -> Message:
    result = await session.execute(select(Message).where(Message.id == message_id))
    message = result.scalar_one_or_none()
    if message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    return message


@router.get("/messages", response_model=list[MessageResponse])
async def list_superadmin_messages(_: None = Depends(_require_secret)):
    await init_user_db(SUPERADMIN_MAILBOX)
    async with get_user_session_factory(SUPERADMIN_MAILBOX)() as session:
        result = await session.execute(select(Message).order_by(Message.created_at.desc()))
        return result.scalars().all()


@router.get("/messages/unread-count", response_model=UnreadCountResponse)
async def superadmin_unread_count(_: None = Depends(_require_secret)):
    await init_user_db(SUPERADMIN_MAILBOX)
    async with get_user_session_factory(SUPERADMIN_MAILBOX)() as session:
        result = await session.execute(
            select(func.count()).select_from(Message).where(Message.read_at.is_(None))
        )
        return UnreadCountResponse(count=result.scalar_one())


@router.post("/messages/read-all", status_code=204)
async def superadmin_mark_all_read(_: None = Depends(_require_secret)):
    await init_user_db(SUPERADMIN_MAILBOX)
    async with get_user_session_factory(SUPERADMIN_MAILBOX)() as session:
        now = datetime.now(timezone.utc)
        result = await session.execute(select(Message).where(Message.read_at.is_(None)))
        for message in result.scalars().all():
            message.read_at = now
        await session.commit()


@router.post("/messages/{message_id}/read", response_model=MessageResponse)
async def superadmin_mark_read(message_id: str, _: None = Depends(_require_secret)):
    await init_user_db(SUPERADMIN_MAILBOX)
    async with get_user_session_factory(SUPERADMIN_MAILBOX)() as session:
        message = await _get_superadmin_message(message_id, session)
        if message.read_at is None:
            message.read_at = datetime.now(timezone.utc)
            await session.commit()
        return message


@router.delete("/messages/{message_id}", status_code=204)
async def superadmin_delete_message(message_id: str, _: None = Depends(_require_secret)):
    await init_user_db(SUPERADMIN_MAILBOX)
    async with get_user_session_factory(SUPERADMIN_MAILBOX)() as session:
        message = await _get_superadmin_message(message_id, session)
        await session.delete(message)  # hard delete
        await session.commit()
