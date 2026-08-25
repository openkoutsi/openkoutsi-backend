import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.core.encryption import EncryptedString
from backend.app.db.base import RegistryBase
from backend.app.db.leases import LeaseMixin


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(RegistryBase):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    # Login identifier for invited/legacy accounts. Nullable because self-serve
    # signup accounts (issue #15) are keyed by email instead; at least one of
    # ``username`` / ``email`` is always present (enforced in the auth layer).
    username: Mapped[Optional[str]] = mapped_column(String, unique=True, nullable=True)
    # Email login identifier for self-serve signup (issue #15). Unique + nullable
    # so many legacy accounts can coexist with a NULL email. ``email_verified_at``
    # is set once the address is confirmed; login-by-email requires it.
    email: Mapped[Optional[str]] = mapped_column(String, unique=True, nullable=True)
    email_verified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    # JSON-encoded list of roles, e.g. '["administrator"]' or '["user"]'.
    roles: Mapped[str] = mapped_column(String, nullable=False, default='["user"]')
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Generation counter for this user's session tokens (issue #102, F-04).
    # Stamped into every access and refresh JWT as ``ver`` and compared on each
    # request, so raising it ends every session the account has open. Session
    # JWTs are otherwise unrevocable: they carry only ``sub``, ``exp`` and
    # ``type``, and nothing on this row could contradict one — a password reset
    # revoked the account's personal access tokens and left the sessions the
    # attacker was already holding untouched, refresh cookie included.
    token_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )

    # Data-processing consent (absorbed from the former DataConsent table).
    consented_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    consent_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    reset_tokens: Mapped[list["PasswordResetToken"]] = relationship(
        "PasswordResetToken", back_populates="user", cascade="all, delete-orphan"
    )
    verification_tokens: Mapped[list["EmailVerificationToken"]] = relationship(
        "EmailVerificationToken", back_populates="user", cascade="all, delete-orphan"
    )
    email_change_tokens: Mapped[list["EmailChangeToken"]] = relationship(
        "EmailChangeToken", back_populates="user", cascade="all, delete-orphan"
    )
    provider_connections: Mapped[list["ProviderConnection"]] = relationship(
        "ProviderConnection", back_populates="user", cascade="all, delete-orphan"
    )
    personal_access_tokens: Mapped[list["PersonalAccessToken"]] = relationship(
        "PersonalAccessToken", back_populates="user", cascade="all, delete-orphan"
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


class EmailVerificationToken(RegistryBase):
    """Single-use email-verification token for self-serve signup (issue #15).

    Mirrors :class:`PasswordResetToken`: only the SHA-256 hash of the raw token is
    stored, tokens are single-use (``used_at``) and expire (``expires_at``).
    """

    __tablename__ = "email_verification_tokens"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id", ondelete="CASCADE"))
    token_hash: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    user: Mapped["User"] = relationship("User", back_populates="verification_tokens")


class EmailChangeToken(RegistryBase):
    """A pending change of an account's address, authorised from *both* ends (issue #62).

    One row is one pending change, carrying two independent secrets: one mailed to
    the address being claimed, one to the address being left. Neither alone moves
    anything — ``users.email`` changes only once every required side is stamped.

    **Why both.** This codebase has no authenticated change-password endpoint: the
    only way to set a password on an existing account is a reset token, and those
    are mailed to ``users.email``. That makes the address the account's sole
    self-serve root of trust, so a one-sided change would let anyone holding just
    the password relocate the recovery channel and then lock the owner out via
    "forgot password" — turning a password leak from recoverable into permanent.
    Requiring the old mailbox costs an attacker exactly what taking the account
    over already costs them, so the feature stops being an escalation.

    ``old_token_hash`` is NULL when the account has no address yet (invite-created
    accounts setting a first one). There is nothing to authorise against in that
    case, so the new side alone completes it; an admin clearing the address is what
    makes a malicious first set undoable.

    It is a separate table from :class:`EmailVerificationToken` because
    :func:`signup` marks *every* unused verification token a user holds as spent
    before issuing a fresh one; a pending change sharing that table would be
    silently voided by an unrelated signup retry.

    ``new_email`` carries no unique constraint. Two users may have a pending change
    to the same address at once — nothing has been claimed until one of them
    finishes, and the loser is turned away at that point by the unique index on
    ``users.email``.
    """

    __tablename__ = "email_change_tokens"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    # Indexed: spent rows are kept rather than deleted, and the lookup for a
    # user's live change runs on every ``GET /auth/account`` — an endpoint the
    # web app polls. ``personal_access_tokens`` indexes its ``user_id`` for the
    # same reason.
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    # Two distinct secrets, each unique across the table. Distinct is the whole
    # point: if one value satisfied both sides, whoever read one mailbox could
    # complete the change alone and the second confirmation would be decoration.
    new_token_hash: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    old_token_hash: Mapped[Optional[str]] = mapped_column(String, unique=True, nullable=True)
    # The address being claimed, stored lowercased. Held here rather than on the
    # user row so an unconfirmed change never touches the login identifier.
    new_email: Mapped[str] = mapped_column(String, nullable=False)
    new_confirmed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    old_confirmed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # Set when the change lands, and when it is cancelled or superseded — in every
    # case meaning "this row can no longer do anything".
    used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    user: Mapped["User"] = relationship("User", back_populates="email_change_tokens")

    @property
    def requires_old_confirmation(self) -> bool:
        """Whether this change needs the outgoing address to authorise it."""
        return self.old_token_hash is not None

    @property
    def fully_confirmed(self) -> bool:
        """Whether every side this change requires has been stamped."""
        if self.new_confirmed_at is None:
            return False
        return self.old_confirmed_at is not None or not self.requires_old_confirmation


class PersonalAccessToken(RegistryBase):
    """A long-lived, scoped, revocable credential a user issues to their own tooling
    (issue #46).

    Mirrors :class:`PasswordResetToken` in shape — only the SHA-256 hash of the
    secret half is stored, and the raw token is shown once at creation — but
    differs in two ways that matter:

    * **It is revocable, and revocation is the point.** ``revoked_at`` is the
      first server-side kill switch for an outstanding credential in this
      codebase; nothing else here can be withdrawn before its own expiry.
    * **Dead rows are kept.** Expiry and revocation end a token's ability to
      authenticate, nothing more. The audit log stores token ids, so deleting
      rows would turn historical entries into unresolvable identifiers at exactly
      the moment somebody is reconstructing what a leaked credential did — and
      keeping ``token_hash`` means a presented-but-revoked token is still
      *recognisable*, so "someone is using a credential we withdrew" stays
      distinguishable from "someone is guessing".

    ``scopes`` is a JSON-encoded list drawn from
    :data:`backend.app.core.scopes.SCOPES`. It, ``name`` and ``expires_at`` are
    fixed at creation: there is no update endpoint, because an editable token
    makes its own audit trail ambiguous.
    """

    __tablename__ = "personal_access_tokens"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token_hash: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    # User-written label. Free text, and revealing on its own often enough
    # ("garmin-sync-for-my-cardiologist") that the admin view never returns it.
    name: Mapped[str] = mapped_column(String, nullable=False)
    # JSON-encoded list of scopes, e.g. '["activities:read","metrics:read"]'.
    scopes: Mapped[str] = mapped_column(String, nullable=False, default="[]")
    # Indexed: the daily expiry sweep filters on it, over a table that only
    # grows because dead rows are retained.
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    # Written coarsely (only when more than an hour stale) — this is a WAL SQLite
    # registry with a pool of 3, and a write on every authenticated request would
    # be the single hottest writer in the system.
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    revoked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # The last expiry-warning stage sent for this token ("expiring_7d",
    # "expiring_1d", "expired"). Without it the daily sweep becomes a daily nag.
    last_expiry_notice: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    user: Mapped["User"] = relationship(
        "User", back_populates="personal_access_tokens"
    )


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
    # Self-serve signup gate (issue #15): when true (and an email provider is
    # configured), anyone can register with their email. Off by default — the
    # instance stays invite-only until an admin flips it.
    allow_self_signup: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )
    # Personal access tokens (issue #46). Unlike the two gates above this
    # defaults **on**: they default off to preserve existing behaviour, and a new
    # feature has none to preserve. A PAT grants strictly less than the session
    # the user already holds — scoped, no admin, no /api/auth, no inbox, no LLM —
    # so it adds duration, not authority. The switch exists for the self-hoster
    # who wants to forbid long-lived credentials on their box, and turning it off
    # refuses *authentication*, not just issuance: tokens handed out beforehand
    # stop working immediately.
    allow_personal_access_tokens: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="1"
    )
    # The MCP tool server (issue #42). Defaults **on**, for the same reason the
    # switch above does: it publishes read-only, scoped tools over data the
    # caller's own credential already reaches, so it adds an interface rather
    # than authority. The switch exists because "an AI client can talk to my
    # training data" is a decision a self-hoster may want to make once, for the
    # whole instance, rather than per token — and because a reverse-proxy rule
    # is not a decision the application can see, test, or report in the admin
    # console. Off refuses the endpoint outright, handshake included.
    allow_mcp_server: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="1"
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
    # Held while one caller is rotating this connection's tokens (issue #50).
    # The rotation is a read-modify-write spanning a network round trip, and
    # Wahoo revokes the old refresh token as soon as the new one is issued — so
    # two callers doing it at once leave one of them holding a dead token and
    # the connection permanently broken. Claiming this column with a conditional
    # UPDATE is what makes exactly one of them the rotator; see
    # ``services.provider_sync.ensure_fresh_token``. NULL (or a time in the past)
    # means free, so a process that dies mid-rotation releases it by expiry
    # rather than wedging the connection.
    refresh_lock_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    user: Mapped["User"] = relationship("User", back_populates="provider_connections")


class RegistryLease(RegistryBase, LeaseMixin):
    """Cross-process mutual exclusion for work that belongs to the *instance*.

    ``SyncLease`` lives in a user's own database because the writes it guards
    land there. This one cannot: what it arbitrates — who runs the background
    pollers this tick — is not any one user's, and a lease is only meaningful to
    holders that can all see the same row. The registry is the only database
    every process opens.

    One name in use: ``background-work``, taken by :mod:`backend.app.services.leadership`.
    Whether the background work runs is one decision, and splitting it per
    poller would let a process be leader for Strava and not for Wahoo — more
    states to reason about, for no benefit.
    """

    __tablename__ = "registry_leases"
