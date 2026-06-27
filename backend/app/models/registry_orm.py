import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.core.encryption import EncryptedString
from backend.app.db.base import RegistryBase


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(RegistryBase):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    username: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    memberships: Mapped[list["TeamMembership"]] = relationship(
        "TeamMembership", back_populates="user", cascade="all, delete-orphan"
    )
    reset_tokens: Mapped[list["PasswordResetToken"]] = relationship(
        "PasswordResetToken", back_populates="user", cascade="all, delete-orphan"
    )
    provider_connections: Mapped[list["ProviderConnection"]] = relationship(
        "ProviderConnection", back_populates="user", cascade="all, delete-orphan"
    )


class PasswordResetToken(RegistryBase):
    __tablename__ = "password_reset_tokens"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id", ondelete="CASCADE"))
    token_hash: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    user: Mapped["User"] = relationship("User", back_populates="reset_tokens")


class Team(RegistryBase):
    __tablename__ = "teams"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    slug: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    # "pending" → awaiting superadmin approval; "active" → normal; "rejected" → blocked
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    # Optional team-level LLM overrides (override global env vars)
    llm_base_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    llm_api_key_enc: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    llm_model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    llm_analysis_context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    memberships: Mapped[list["TeamMembership"]] = relationship(
        "TeamMembership", back_populates="team", cascade="all, delete-orphan"
    )
    invitations: Mapped[list["Invitation"]] = relationship(
        "Invitation", back_populates="team", cascade="all, delete-orphan"
    )


class TeamMembership(RegistryBase):
    __tablename__ = "team_memberships"
    __table_args__ = (UniqueConstraint("team_id", "user_id"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    team_id: Mapped[str] = mapped_column(String, ForeignKey("teams.id", ondelete="CASCADE"))
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id", ondelete="CASCADE"))
    # JSON-encoded list of roles, e.g. '["administrator","coach"]'
    roles: Mapped[str] = mapped_column(String, nullable=False, default='["user"]')
    joined_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    team: Mapped["Team"] = relationship("Team", back_populates="memberships")
    user: Mapped["User"] = relationship("User", back_populates="memberships")


class Invitation(RegistryBase):
    __tablename__ = "invitations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    team_id: Mapped[str] = mapped_column(String, ForeignKey("teams.id", ondelete="CASCADE"))
    token_hash: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    # JSON-encoded default roles for the invitee, e.g. '["user"]'
    roles: Mapped[str] = mapped_column(String, nullable=False, default='["user"]')
    created_by_user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE")
    )
    used_by_user_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    note: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    team: Mapped["Team"] = relationship("Team", back_populates="invitations")


class JoinRequest(RegistryBase):
    """A self-serve request from a person to join a team.

    Credentials are captured up front (hashed) so that, on admin approval, the
    user account + membership can be created without further interaction.
    """

    __tablename__ = "join_requests"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    team_id: Mapped[str] = mapped_column(String, ForeignKey("teams.id", ondelete="CASCADE"))
    username: Mapped[str] = mapped_column(String, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    display_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # "pending" → awaiting decision; "approved"; "rejected"
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    decided_by_user_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)


class DataConsent(RegistryBase):
    """Records that a user has accepted the data processing terms for a team."""

    __tablename__ = "data_consents"
    __table_args__ = (UniqueConstraint("user_id", "team_id"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id", ondelete="CASCADE"))
    team_id: Mapped[str] = mapped_column(String, ForeignKey("teams.id", ondelete="CASCADE"))
    consented_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    consent_version: Mapped[str] = mapped_column(String, nullable=False, default="1.0")


class ProviderConnection(RegistryBase):
    """OAuth connections belong to the user globally, not to a specific team.

    A user connects Strava once; synced activities are written to all teams
    they belong to.
    """

    __tablename__ = "provider_connections"
    __table_args__ = (UniqueConstraint("user_id", "provider"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id", ondelete="CASCADE"))
    provider: Mapped[str] = mapped_column(String, nullable=False)  # "strava", "wahoo", …
    provider_athlete_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    access_token: Mapped[Optional[str]] = mapped_column(EncryptedString, nullable=True)
    refresh_token: Mapped[Optional[str]] = mapped_column(EncryptedString, nullable=True)
    token_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    scopes: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    user: Mapped["User"] = relationship("User", back_populates="provider_connections")
