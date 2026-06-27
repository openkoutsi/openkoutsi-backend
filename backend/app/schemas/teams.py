from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class TeamResponse(BaseModel):
    id: str
    slug: str
    name: str
    created_at: datetime


class MemberResponse(BaseModel):
    user_id: str
    username: str
    roles: list[str]
    joined_at: datetime
    consented_at: Optional[datetime] = None
    consent_version: Optional[str] = None


class ConsentRequest(BaseModel):
    consent_version: str = "1.0"


class ConsentResponse(BaseModel):
    consented_at: datetime
    consent_version: str


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


class MemberRolesUpdate(BaseModel):
    roles: list[str]


class PasswordResetLinkResponse(BaseModel):
    reset_url: str


class TeamSettingsResponse(BaseModel):
    llm_base_url: Optional[str]
    llm_model: Optional[str]
    llm_api_key_set: bool
    llm_analysis_context: Optional[str]


class TeamSettingsPatch(BaseModel):
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    clear_llm_api_key: bool = False
    llm_analysis_context: Optional[str] = None


class TeamSignupRequest(BaseModel):
    team_name: str
    slug: str
    admin_username: str
    admin_password: str
    admin_display_name: Optional[str] = None

    from pydantic import field_validator

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

    @field_validator("slug")
    @classmethod
    def slug_format(cls, v: str) -> str:
        import re
        if not re.match(r"^[a-z0-9][a-z0-9-]{0,61}[a-z0-9]$", v):
            raise ValueError(
                "Slug must be lowercase letters, digits, and hyphens; "
                "2-63 characters; cannot start or end with a hyphen"
            )
        return v


class TeamSignupResponse(BaseModel):
    id: str
    slug: str
    name: str
    status: str
    created_at: datetime


class SuperadminTeamResponse(BaseModel):
    id: str
    slug: str
    name: str
    status: str
    created_at: datetime
    member_count: int
    consented_count: int = 0


class SuperadminUserTeam(BaseModel):
    team_id: str
    team_slug: str
    team_name: str
    roles: list[str]
    joined_at: datetime
    consented_at: Optional[datetime] = None
    consent_version: Optional[str] = None


class SuperadminUserResponse(BaseModel):
    id: str
    username: str
    created_at: datetime
    teams: list[SuperadminUserTeam]


class SetupStatusResponse(BaseModel):
    needs_setup: bool


class SetupRequest(BaseModel):
    team_name: str
    slug: str
    admin_username: str
    admin_password: str
    admin_display_name: Optional[str] = None

    from pydantic import field_validator

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

    @field_validator("slug")
    @classmethod
    def slug_format(cls, v: str) -> str:
        import re
        if not re.match(r"^[a-z0-9][a-z0-9-]{0,61}[a-z0-9]$", v):
            raise ValueError(
                "Slug must be lowercase letters, digits, and hyphens; "
                "2-63 characters; cannot start or end with a hyphen"
            )
        return v
