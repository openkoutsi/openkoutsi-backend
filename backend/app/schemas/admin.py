from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, field_validator


# ── First-run setup ────────────────────────────────────────────────────────

class SetupStatusResponse(BaseModel):
    needs_setup: bool


class SetupRequest(BaseModel):
    """Create the first instance administrator. No team — single-instance."""
    admin_username: str
    admin_password: str
    admin_display_name: Optional[str] = None

    @field_validator("admin_password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 12:
            raise ValueError("Password must be at least 12 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


# ── Users (instance admin) ─────────────────────────────────────────────────

class UserResponse(BaseModel):
    id: str
    username: str
    roles: list[str]
    created_at: datetime
    consented_at: Optional[datetime] = None
    consent_version: Optional[str] = None


class UserRolesUpdate(BaseModel):
    roles: list[str]


class PasswordResetLinkResponse(BaseModel):
    reset_url: str


# ── Consent ────────────────────────────────────────────────────────────────

class ConsentRequest(BaseModel):
    consent_version: str = "1.0"


class ConsentResponse(BaseModel):
    consented_at: datetime
    consent_version: str


# ── Invitations (instance admin) ───────────────────────────────────────────

class InvitationCreate(BaseModel):
    roles: list[str] = ["user"]
    expires_in_days: Optional[int] = 7
    note: Optional[str] = None


class InvitationResponse(BaseModel):
    id: str
    roles: list[str]
    note: Optional[str]
    created_by_username: str
    used_by_username: Optional[str]
    expires_at: Optional[datetime]
    used_at: Optional[datetime]
    created_at: datetime
    url: Optional[str] = None


# ── Instance settings (instance admin) ─────────────────────────────────────

class LlmModelConfigIn(BaseModel):
    """A selectable model *preset* — a full or partial connection.

    A preset is a self-contained connection: ``base_url`` / ``model`` /
    ``api_key`` / ``headers`` / ``body``. This lets an admin offer distinct
    providers (Anthropic, Mistral, …) as presets that a user picks between; the
    **first preset in the list is the instance default**. ``api_key`` is
    write-only; omit it to keep the stored key, or set ``api_key_clear`` to
    remove it. ``name`` is the stable internal identifier (what a user's
    selection is stored as); ``label`` is the human-friendly name shown to users
    (defaults to ``name``).
    """
    name: str
    label: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    api_key_clear: bool = False
    headers: dict[str, str] = {}
    body: dict[str, Any] = {}

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Model name must not be blank")
        return v.strip()


class LlmModelConfigOut(BaseModel):
    """A selectable model preset as returned to the admin (no secret leaked)."""
    name: str
    label: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    api_key_set: bool = False
    headers: dict[str, str] = {}
    body: dict[str, Any] = {}


class InstanceSettingsResponse(BaseModel):
    llm_analysis_context: Optional[str]
    admin_contact: Optional[str]
    # The instance's entire LLM config: selectable presets, first = default.
    llm_models: list[LlmModelConfigOut] = []


class InstanceSettingsPatch(BaseModel):
    llm_analysis_context: Optional[str] = None
    admin_contact: Optional[str] = None
    # Full-replacement list: send the complete desired preset list (first entry
    # is the default), or omit to leave unchanged.
    llm_models: Optional[list[LlmModelConfigIn]] = None
