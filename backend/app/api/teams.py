import hashlib
import json
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import TeamContext, get_current_user, hash_password
from backend.app.core.config import settings
from backend.app.core.file_encryption import decrypt_team_secret, encrypt_team_secret
from backend.app.core.limiter import limiter
from backend.app.db.registry import get_registry_session
from backend.app.models.registry_orm import DataConsent, Invitation, PasswordResetToken, Team, TeamMembership, User
from backend.app.schemas.teams import (
    InvitationCreate,
    InvitationResponse,
    MemberResponse,
    MemberRolesUpdate,
    PasswordResetLinkResponse,
    TeamSettingsPatch,
    TeamSettingsResponse,
)

router = APIRouter(prefix="/teams/{slug}", tags=["teams"])


async def _resolve_team(slug: str, session: AsyncSession) -> Team:
    result = await session.execute(select(Team).where(Team.slug == slug))
    team = result.scalar_one_or_none()
    if team is None:
        raise HTTPException(status_code=404, detail="Team not found")
    return team


def _require_admin(ctx: TeamContext, team: Team) -> None:
    if ctx.team_id != team.id or not ctx.is_admin:
        raise HTTPException(status_code=403, detail="Administrator role required")


def _require_admin_or_coach(ctx: TeamContext, team: Team) -> None:
    if ctx.team_id != team.id or not (ctx.is_admin or ctx.is_coach):
        raise HTTPException(status_code=403, detail="Coach or administrator role required")


# ── Members ──────────────────────────────────────────────────────────────────

@router.get("/members", response_model=list[MemberResponse])
async def list_members(
    slug: str,
    ctx: TeamContext = Depends(get_current_user),
    session: AsyncSession = Depends(get_registry_session),
):
    team = await _resolve_team(slug, session)
    _require_admin_or_coach(ctx, team)

    result = await session.execute(
        select(TeamMembership, User)
        .join(User, User.id == TeamMembership.user_id)
        .where(TeamMembership.team_id == team.id)
        .order_by(TeamMembership.joined_at)
    )
    rows = result.all()

    consent_result = await session.execute(
        select(DataConsent).where(DataConsent.team_id == team.id)
    )
    consents = {c.user_id: c for c in consent_result.scalars().all()}

    return [
        MemberResponse(
            user_id=user.id,
            username=user.username,
            roles=json.loads(membership.roles),
            joined_at=membership.joined_at,
            consented_at=consents[user.id].consented_at if user.id in consents else None,
            consent_version=consents[user.id].consent_version if user.id in consents else None,
        )
        for membership, user in rows
    ]


@router.patch("/members/{user_id}/roles", response_model=MemberResponse)
async def update_member_roles(
    slug: str,
    user_id: str,
    body: MemberRolesUpdate,
    ctx: TeamContext = Depends(get_current_user),
    session: AsyncSession = Depends(get_registry_session),
):
    team = await _resolve_team(slug, session)
    _require_admin(ctx, team)

    result = await session.execute(
        select(TeamMembership, User)
        .join(User, User.id == TeamMembership.user_id)
        .where(TeamMembership.team_id == team.id, TeamMembership.user_id == user_id)
    )
    row = result.one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Member not found")
    membership, user = row

    valid_roles = {"administrator", "coach", "user"}
    invalid = set(body.roles) - valid_roles
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid roles: {invalid}")

    membership.roles = json.dumps(body.roles)
    await session.commit()
    return MemberResponse(
        user_id=user.id,
        username=user.username,
        roles=body.roles,
        joined_at=membership.joined_at,
    )


@router.delete("/members/{user_id}", status_code=204)
async def remove_member(
    slug: str,
    user_id: str,
    ctx: TeamContext = Depends(get_current_user),
    session: AsyncSession = Depends(get_registry_session),
):
    team = await _resolve_team(slug, session)
    _require_admin(ctx, team)

    result = await session.execute(
        select(TeamMembership).where(
            TeamMembership.team_id == team.id,
            TeamMembership.user_id == user_id,
        )
    )
    membership = result.scalar_one_or_none()
    if membership is None:
        raise HTTPException(status_code=404, detail="Member not found")
    await session.delete(membership)
    await session.commit()


@router.post("/members/{user_id}/password-reset", response_model=PasswordResetLinkResponse)
@limiter.limit("10/hour")
async def generate_member_password_reset(
    request: Request,
    slug: str,
    user_id: str,
    ctx: TeamContext = Depends(get_current_user),
    session: AsyncSession = Depends(get_registry_session),
):
    team = await _resolve_team(slug, session)
    _require_admin(ctx, team)

    # Verify the user is a member of this team
    mb_result = await session.execute(
        select(TeamMembership).where(
            TeamMembership.team_id == team.id,
            TeamMembership.user_id == user_id,
        )
    )
    if mb_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Member not found")

    user_result = await session.execute(
        select(User).where(User.id == user_id, User.deleted_at.is_(None))
    )
    user = user_result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Invalidate any existing unused tokens for this user
    existing = await session.execute(
        select(PasswordResetToken).where(
            PasswordResetToken.user_id == user.id,
            PasswordResetToken.used_at.is_(None),
        )
    )
    for token_row in existing.scalars():
        token_row.used_at = datetime.now(timezone.utc)

    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

    session.add(PasswordResetToken(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=expires_at,
    ))
    await session.commit()

    reset_url = f"{settings.frontend_url}/t/{slug}/reset-password?token={token}"
    return PasswordResetLinkResponse(reset_url=reset_url)


# ── Invitations ───────────────────────────────────────────────────────────────

@router.post("/invitations", response_model=InvitationResponse, status_code=201)
@limiter.limit("30/hour")
async def create_invitation(
    request: Request,
    slug: str,
    body: InvitationCreate,
    ctx: TeamContext = Depends(get_current_user),
    session: AsyncSession = Depends(get_registry_session),
):
    team = await _resolve_team(slug, session)
    _require_admin(ctx, team)

    valid_roles = {"administrator", "coach", "user"}
    invalid = set(body.roles) - valid_roles
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid roles: {invalid}")

    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()

    expires_at = None
    if body.expires_in_days is not None:
        expires_at = datetime.now(timezone.utc) + timedelta(days=body.expires_in_days)

    # Look up creator's username for the response
    creator_result = await session.execute(select(User).where(User.id == ctx.user_id))
    creator = creator_result.scalar_one()

    invitation = Invitation(
        id=str(uuid.uuid4()),
        team_id=team.id,
        token_hash=token_hash,
        roles=json.dumps(body.roles),
        note=body.note or None,
        created_by_user_id=ctx.user_id,
        expires_at=expires_at,
    )
    session.add(invitation)
    await session.commit()

    invite_url = f"{settings.frontend_url}/t/{slug}/register?token={raw_token}"
    return InvitationResponse(
        id=invitation.id,
        roles=body.roles,
        note=invitation.note,
        created_by_username=creator.username,
        used_by_username=None,
        expires_at=expires_at,
        used_at=None,
        created_at=invitation.created_at,
        url=invite_url,
    )


@router.get("/invitations", response_model=list[InvitationResponse])
async def list_invitations(
    slug: str,
    ctx: TeamContext = Depends(get_current_user),
    session: AsyncSession = Depends(get_registry_session),
):
    team = await _resolve_team(slug, session)
    _require_admin(ctx, team)

    result = await session.execute(
        select(Invitation).where(Invitation.team_id == team.id).order_by(Invitation.created_at.desc())
    )
    invitations = result.scalars().all()

    # Collect user IDs for username lookup
    user_ids = set()
    for inv in invitations:
        user_ids.add(inv.created_by_user_id)
        if inv.used_by_user_id:
            user_ids.add(inv.used_by_user_id)

    users_result = await session.execute(select(User).where(User.id.in_(user_ids)))
    users_by_id = {u.id: u for u in users_result.scalars()}

    return [
        InvitationResponse(
            id=inv.id,
            roles=json.loads(inv.roles),
            note=inv.note,
            created_by_username=users_by_id[inv.created_by_user_id].username if inv.created_by_user_id in users_by_id else "(deleted)",
            used_by_username=(
                users_by_id[inv.used_by_user_id].username if inv.used_by_user_id and inv.used_by_user_id in users_by_id else None
            ),
            expires_at=inv.expires_at,
            used_at=inv.used_at,
            created_at=inv.created_at,
        )
        for inv in invitations
    ]


@router.delete("/invitations/{invitation_id}", status_code=204)
async def revoke_invitation(
    slug: str,
    invitation_id: str,
    ctx: TeamContext = Depends(get_current_user),
    session: AsyncSession = Depends(get_registry_session),
):
    team = await _resolve_team(slug, session)
    _require_admin(ctx, team)

    result = await session.execute(
        select(Invitation).where(
            Invitation.id == invitation_id,
            Invitation.team_id == team.id,
        )
    )
    invitation = result.scalar_one_or_none()
    if invitation is None:
        raise HTTPException(status_code=404, detail="Invitation not found")
    if invitation.used_at is not None:
        raise HTTPException(status_code=400, detail="Cannot revoke an already-used invitation")
    await session.delete(invitation)
    await session.commit()


# ── Team settings (LLM config) ────────────────────────────────────────────────

@router.get("/settings", response_model=TeamSettingsResponse)
async def get_team_settings(
    slug: str,
    ctx: TeamContext = Depends(get_current_user),
    session: AsyncSession = Depends(get_registry_session),
):
    team = await _resolve_team(slug, session)
    _require_admin(ctx, team)
    return TeamSettingsResponse(
        llm_base_url=team.llm_base_url,
        llm_model=team.llm_model,
        llm_api_key_set=team.llm_api_key_enc is not None,
        llm_analysis_context=team.llm_analysis_context,
    )


@router.patch("/settings", response_model=TeamSettingsResponse)
async def update_team_settings(
    slug: str,
    body: TeamSettingsPatch,
    ctx: TeamContext = Depends(get_current_user),
    session: AsyncSession = Depends(get_registry_session),
):
    team = await _resolve_team(slug, session)
    _require_admin(ctx, team)

    if body.llm_base_url is not None:
        team.llm_base_url = body.llm_base_url or None
    if body.llm_model is not None:
        team.llm_model = body.llm_model or None
    if body.llm_analysis_context is not None:
        team.llm_analysis_context = body.llm_analysis_context or None

    if body.clear_llm_api_key:
        team.llm_api_key_enc = None
    elif body.llm_api_key:
        if not settings.encryption_key:
            raise HTTPException(
                status_code=400,
                detail="ENCRYPTION_KEY not set — cannot store encrypted API key",
            )
        team.llm_api_key_enc = encrypt_team_secret(body.llm_api_key, team.id)

    await session.commit()
    return TeamSettingsResponse(
        llm_base_url=team.llm_base_url,
        llm_model=team.llm_model,
        llm_api_key_set=team.llm_api_key_enc is not None,
        llm_analysis_context=team.llm_analysis_context,
    )
