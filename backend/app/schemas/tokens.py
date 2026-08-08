"""Request/response models for personal access tokens (issue #46)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.app.services.personal_access_tokens import (
    DEFAULT_LIFETIME_DAYS,
    MAX_LIFETIME_DAYS,
)


class ScopeInfo(BaseModel):
    """One entry of the scope vocabulary, for the creation UI."""

    name: str
    description: str
    # Presented apart from the ordinary read scopes: `athlete:export` returns the
    # whole record in one call and deserves its own deliberate tick.
    sensitive: bool = False


class TokenScopesResponse(BaseModel):
    scopes: list[ScopeInfo]
    allowed_lifetime_days: list[int]
    default_lifetime_days: int = DEFAULT_LIFETIME_DAYS
    max_lifetime_days: int = MAX_LIFETIME_DAYS


class PersonalAccessTokenCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    scopes: list[str] = Field(default_factory=list)
    # No "never". The ceiling is enforced server-side as well as in the picker,
    # so a hand-rolled POST asking for ten years is rejected.
    expires_in_days: int = Field(
        default=DEFAULT_LIFETIME_DAYS, ge=1, le=MAX_LIFETIME_DAYS
    )


class PersonalAccessTokenResponse(BaseModel):
    """A token's metadata. Never carries the secret or its hash."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    scopes: list[str]
    status: str  # "active" | "expired" | "revoked"
    expires_at: datetime
    last_used_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    created_at: datetime


class PersonalAccessTokenCreated(PersonalAccessTokenResponse):
    """The create response — the only time the secret is ever returned."""

    token: str


class AdminPersonalAccessTokenResponse(BaseModel):
    """An administrator's view of one user's token.

    Metadata only, and deliberately **no name**: token names are user-written
    free text and revealing on their own ("garmin-sync-for-my-cardiologist").
    Revocation needs the id, not the label.
    """

    id: str
    scopes: list[str]
    status: str
    expires_at: datetime
    last_used_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    created_at: datetime
