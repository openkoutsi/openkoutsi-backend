import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, String, Text, UniqueConstraint
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
    # JSON-encoded list of roles, e.g. '["administrator"]' or '["user"]'.
    roles: Mapped[str] = mapped_column(String, nullable=False, default='["user"]')
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Data-processing consent (absorbed from the former DataConsent table).
    consented_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    consent_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)

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


class Invitation(RegistryBase):
    """An instance-wide invitation issued by an administrator.

    Onboarding is invite-only: registration requires a valid invite token.
    """

    __tablename__ = "invitations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
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


class InstanceSettings(RegistryBase):
    """Single-row table holding instance-wide settings.

    Replaces the former per-team LLM overrides; managed by an instance admin.
    The row uses a fixed primary key so there is always at most one.
    """

    __tablename__ = "instance_settings"

    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    # Curated list of selectable presets — the instance's entire LLM config.
    # The **first entry is the instance default** selection. Each entry is a
    # self-contained connection (there are no instance single-config or env-var
    # fallbacks; an omitted field is simply absent):
    #   ``{"name": str,            # stable identifier / selection value
    #      "label": str | None,    # human-friendly display name
    #      "base_url": str | None,
    #      "model": str | None,    # upstream model id (defaults to name)
    #      "api_key_enc": str | None,  # encrypted per-preset key
    #      "headers": {<extra request headers>},
    #      "body": {<extra chat-completion body params>}}``
    llm_models: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    llm_analysis_context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    admin_contact: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # Opt-in gate (issue #9): when true, only users with an active LLM-access
    # entitlement may use the instance's LLM credentials. Everyone else can still
    # use LLM features via BYOK, or gets a machine-readable "subscription
    # required" error. Defaults to off — behaviour is unchanged until flipped.
    llm_requires_subscription: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class LlmEntitlement(RegistryBase):
    """Per-user "LLM access" entitlement (issue #9).

    A table rather than a role: entitlements carry expiry, provenance and audit
    fields that a plain role JSON list can't, and the table is an idempotent
    upsert target for a future payment handler (#16). Roles keep meaning
    *permissions*; entitlements mean *commercial state*.

    Phase 1 grants are ``source="manual"`` (admin console). Phase 2 will use a
    free-form payment-provider slug (``stripe``, ``paddle``, …) — not an enum.

    Entitled predicate::

        status == "active" and starts_at <= now and (expires_at is None or expires_at > now)
    """

    __tablename__ = "llm_entitlements"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")
    source: Mapped[str] = mapped_column(String, nullable=False, default="manual")
    granted_by_user_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    starts_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    external_ref: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class ProviderConnection(RegistryBase):
    """OAuth connections belong to the user globally.

    A user connects Strava once; synced activities are written to their own DB.
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
