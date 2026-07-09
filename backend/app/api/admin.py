"""Instance-admin API.

Consolidates the former per-team `teams.py` + `members.py` + `superadmin.py`
into a single token-scoped resource. There is one instance and a single global
``administrator`` role; these endpoints manage users, instance-wide invitations,
and instance LLM settings.
"""
import hashlib
import json
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import UserContext, get_current_user
from backend.app.core.config import settings
from backend.app.core.file_encryption import encrypt_instance_secret
from backend.app.core.limiter import limiter
from backend.app.db.registry import get_registry_session
from backend.app.db.user_session import delete_user_db
from backend.app.models.registry_orm import (
    InstanceSettings,
    Invitation,
    PasswordResetToken,
    ProviderConnection,
    User,
)
from backend.app.schemas.admin import (
    InstanceSettingsPatch,
    InstanceSettingsResponse,
    InvitationCreate,
    InvitationResponse,
    LlmModelConfigIn,
    LlmModelConfigOut,
    PasswordResetLinkResponse,
    UserResponse,
    UserRolesUpdate,
)
from backend.app.schemas.pagination import Page, PageParams, paginate_params

router = APIRouter(prefix="/admin", tags=["admin"])

VALID_ROLES = {"administrator", "user"}


async def require_admin(
    ctx: UserContext = Depends(get_current_user),
) -> UserContext:
    if not ctx.is_admin:
        raise HTTPException(status_code=403, detail="Administrator role required")
    return ctx


def _roles_of(user: User) -> list[str]:
    try:
        return json.loads(user.roles) if user.roles else []
    except (TypeError, ValueError):
        return []


def _user_response(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        username=user.username,
        roles=_roles_of(user),
        created_at=user.created_at,
        consented_at=user.consented_at,
        consent_version=user.consent_version,
    )


# ── Users ──────────────────────────────────────────────────────────────────

@router.get("/users", response_model=Page[UserResponse],
            operation_id="listUsers", summary="List users")
async def list_users(
    _: UserContext = Depends(require_admin),
    params: PageParams = Depends(paginate_params),
    session: AsyncSession = Depends(get_registry_session),
):
    total = (await session.execute(
        select(func.count()).select_from(User).where(User.deleted_at.is_(None))
    )).scalar_one()
    result = await session.execute(
        select(User)
        .where(User.deleted_at.is_(None))
        .order_by(User.created_at)
        .offset(params.offset)
        .limit(params.page_size)
    )
    items = [_user_response(u) for u in result.scalars().all()]
    return Page.build(items, total, params.page, params.page_size)


@router.patch("/users/{user_id}/roles", response_model=UserResponse,
              operation_id="updateUserRoles", summary="Update a user's roles")
async def update_user_roles(
    user_id: str,
    body: UserRolesUpdate,
    _: UserContext = Depends(require_admin),
    session: AsyncSession = Depends(get_registry_session),
):
    invalid = set(body.roles) - VALID_ROLES
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid roles: {invalid}")

    result = await session.execute(
        select(User).where(User.id == user_id, User.deleted_at.is_(None))
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    user.roles = json.dumps(body.roles)
    await session.commit()
    await session.refresh(user)
    return _user_response(user)


@router.delete("/users/{user_id}", status_code=204,
               operation_id="deleteUser", summary="Delete a user")
async def delete_user(
    user_id: str,
    ctx: UserContext = Depends(require_admin),
    session: AsyncSession = Depends(get_registry_session),
):
    if user_id == ctx.user_id:
        raise HTTPException(status_code=400, detail="Admins cannot delete their own account here")

    result = await session.execute(
        select(User).where(User.id == user_id, User.deleted_at.is_(None))
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    await session.delete(user)
    await session.commit()
    try:
        await delete_user_db(user_id)
    except Exception:
        pass


@router.post("/users/{user_id}/password-reset", response_model=PasswordResetLinkResponse,
             operation_id="createUserPasswordResetLink",
             summary="Generate a password-reset link for a user")
@limiter.limit("10/hour")
async def generate_user_password_reset(
    request: Request,
    user_id: str,
    _: UserContext = Depends(require_admin),
    session: AsyncSession = Depends(get_registry_session),
):
    user_result = await session.execute(
        select(User).where(User.id == user_id, User.deleted_at.is_(None))
    )
    user = user_result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

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

    reset_url = f"{settings.frontend_url}/reset-password?token={token}"
    return PasswordResetLinkResponse(reset_url=reset_url)


# ── Invitations ────────────────────────────────────────────────────────────

@router.post("/invitations", response_model=InvitationResponse, status_code=201,
             operation_id="createInvitation", summary="Create an invitation")
@limiter.limit("30/hour")
async def create_invitation(
    request: Request,
    body: InvitationCreate,
    ctx: UserContext = Depends(require_admin),
    session: AsyncSession = Depends(get_registry_session),
):
    invalid = set(body.roles) - VALID_ROLES
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid roles: {invalid}")

    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()

    expires_at = None
    if body.expires_in_days is not None:
        expires_at = datetime.now(timezone.utc) + timedelta(days=body.expires_in_days)

    creator_result = await session.execute(select(User).where(User.id == ctx.user_id))
    creator = creator_result.scalar_one()

    invitation = Invitation(
        id=str(uuid.uuid4()),
        token_hash=token_hash,
        roles=json.dumps(body.roles),
        note=body.note or None,
        created_by_user_id=ctx.user_id,
        expires_at=expires_at,
    )
    session.add(invitation)
    await session.commit()

    invite_url = f"{settings.frontend_url}/register?token={raw_token}"
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


@router.get("/invitations", response_model=Page[InvitationResponse],
            operation_id="listInvitations", summary="List invitations")
async def list_invitations(
    _: UserContext = Depends(require_admin),
    params: PageParams = Depends(paginate_params),
    session: AsyncSession = Depends(get_registry_session),
):
    total = (await session.execute(
        select(func.count()).select_from(Invitation)
    )).scalar_one()
    result = await session.execute(
        select(Invitation)
        .order_by(Invitation.created_at.desc())
        .offset(params.offset)
        .limit(params.page_size)
    )
    invitations = result.scalars().all()

    user_ids: set[str] = set()
    for inv in invitations:
        user_ids.add(inv.created_by_user_id)
        if inv.used_by_user_id:
            user_ids.add(inv.used_by_user_id)

    users_result = await session.execute(select(User).where(User.id.in_(user_ids)))
    users_by_id = {u.id: u for u in users_result.scalars()}

    items = [
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
    return Page.build(items, total, params.page, params.page_size)


@router.delete("/invitations/{invitation_id}", status_code=204,
               operation_id="revokeInvitation", summary="Revoke an invitation")
async def revoke_invitation(
    invitation_id: str,
    _: UserContext = Depends(require_admin),
    session: AsyncSession = Depends(get_registry_session),
):
    result = await session.execute(
        select(Invitation).where(Invitation.id == invitation_id)
    )
    invitation = result.scalar_one_or_none()
    if invitation is None:
        raise HTTPException(status_code=404, detail="Invitation not found")
    if invitation.used_at is not None:
        raise HTTPException(status_code=400, detail="Cannot revoke an already-used invitation")
    await session.delete(invitation)
    await session.commit()


# ── Instance settings (LLM config) ─────────────────────────────────────────

async def _get_or_create_settings(session: AsyncSession) -> InstanceSettings:
    result = await session.execute(select(InstanceSettings).limit(1))
    instance = result.scalar_one_or_none()
    if instance is None:
        instance = InstanceSettings(id=1)
        session.add(instance)
        await session.flush()
    return instance


def _build_presets(
    incoming: list[LlmModelConfigIn],
    existing: list | None,
) -> list[dict]:
    """Build the stored preset list, encrypting/preserving per-preset API keys.

    ``llm_models`` is a full-replacement list. For each preset: a supplied
    ``api_key`` is encrypted; ``api_key_clear`` drops it; otherwise the key
    stored for a preset with the same name is preserved so editing other fields
    doesn't require re-entering the key.
    """
    prior = {
        str(e.get("name", "")): e
        for e in (existing or [])
        if isinstance(e, dict) and e.get("name")
    }
    out: list[dict] = []
    for m in incoming:
        entry: dict = {"name": m.name}
        if m.label and m.label.strip():
            entry["label"] = m.label.strip()
        if m.base_url and m.base_url.strip():
            entry["base_url"] = m.base_url.strip()
        if m.model and m.model.strip():
            entry["model"] = m.model.strip()
        headers = {k: v for k, v in (m.headers or {}).items() if k.strip()}
        if headers:
            entry["headers"] = headers
        if m.body:
            entry["body"] = m.body

        if m.api_key_clear:
            pass  # explicitly drop any stored key
        elif m.api_key:
            if not settings.encryption_key:
                raise HTTPException(
                    status_code=400,
                    detail="ENCRYPTION_KEY not set — cannot store encrypted API key",
                )
            entry["api_key_enc"] = encrypt_instance_secret(m.api_key)
        else:
            prev = prior.get(m.name)
            if prev and prev.get("api_key_enc"):
                entry["api_key_enc"] = prev["api_key_enc"]

        out.append(entry)
    return out


def _preset_out(entry: dict) -> LlmModelConfigOut:
    """Map a stored preset entry to its API form, hiding the encrypted key."""
    return LlmModelConfigOut(
        name=str(entry.get("name", "")),
        label=entry.get("label"),
        base_url=entry.get("base_url"),
        model=entry.get("model"),
        api_key_set=bool(entry.get("api_key_enc")),
        headers=entry.get("headers") or {},
        body=entry.get("body") or {},
    )


def _settings_response(instance: InstanceSettings) -> InstanceSettingsResponse:
    return InstanceSettingsResponse(
        llm_analysis_context=instance.llm_analysis_context,
        admin_contact=instance.admin_contact,
        llm_models=[_preset_out(e) for e in (instance.llm_models or []) if isinstance(e, dict)],
    )


@router.get("/settings", response_model=InstanceSettingsResponse,
            operation_id="getInstanceSettings", summary="Get instance settings")
async def get_instance_settings(
    _: UserContext = Depends(require_admin),
    session: AsyncSession = Depends(get_registry_session),
):
    instance = await _get_or_create_settings(session)
    return _settings_response(instance)


@router.patch("/settings", response_model=InstanceSettingsResponse,
              operation_id="updateInstanceSettings", summary="Update instance settings")
async def update_instance_settings(
    body: InstanceSettingsPatch,
    _: UserContext = Depends(require_admin),
    session: AsyncSession = Depends(get_registry_session),
):
    instance = await _get_or_create_settings(session)

    if body.llm_analysis_context is not None:
        instance.llm_analysis_context = body.llm_analysis_context or None
    if body.admin_contact is not None:
        instance.admin_contact = body.admin_contact or None
    if body.llm_models is not None:
        instance.llm_models = _build_presets(body.llm_models, instance.llm_models) or None

    await session.commit()
    await session.refresh(instance)
    return _settings_response(instance)
