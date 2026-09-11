from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

from backend.app.schemas.auth import _validate_password_strength


# ── First-run setup ────────────────────────────────────────────────────────

class SetupStatusResponse(BaseModel):
    needs_setup: bool


class SetupRequest(BaseModel):
    """Create the first instance administrator. No team — single-instance."""
    admin_username: str
    admin_password: str
    admin_display_name: Optional[str] = None

    # Shares the rules with every other password field rather than restating
    # them. The copy that used to live here had drifted out of step by omission
    # — it never gained the maximum length, so the setup wizard answered a long
    # passphrase with a 500 (issue #102, F-07).
    @field_validator("admin_password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        return _validate_password_strength(v)


# ── Users (instance admin) ─────────────────────────────────────────────────

class LlmEntitlementSummary(BaseModel):
    """A user's LLM-access entitlement as shown in the admin console (issue #9)."""
    status: str
    active: bool  # the entitled predicate (status + start/expiry window) at read time
    source: str
    starts_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    notes: Optional[str] = None
    updated_at: Optional[datetime] = None


class UserResponse(BaseModel):
    id: str
    # Nullable since self-serve signup accounts are keyed by email, not username.
    username: Optional[str] = None
    email: Optional[str] = None
    # When the address above was confirmed; null while it is unconfirmed. Read
    # it together with ``email``: null on an account that has no address means
    # "nothing to confirm", not "unconfirmed". A self-serve signup writes the
    # user row before the address is confirmed, so an abandoned attempt leaves a
    # row here that can never sign in — worth telling apart from a real account
    # in the console, and the only place an admin can see it at all.
    email_verified_at: Optional[datetime] = None
    roles: list[str]
    created_at: datetime
    consented_at: Optional[datetime] = None
    consent_version: Optional[str] = None
    # Null when the user has never been granted an entitlement.
    llm_entitlement: Optional[LlmEntitlementSummary] = None


class UserRolesUpdate(BaseModel):
    roles: list[str]


class UserEmailUpdate(BaseModel):
    """Admin set/clear of a user's address (issue #62).

    ``None`` clears it — the escape hatch for an address its owner can no longer
    reach, for which the only previous remedy was deleting the user and their
    training data.

    Required, not defaulted, though still nullable: with ``= None`` both an empty
    body and one misnaming the field would read as "clear it", and this clear is
    destructive (it drops the login identifier, ends every session and revokes
    every token). ``new_email`` is what the user-facing endpoint calls it, so the
    slip is easy to make. No default turns both into a 422 while leaving a
    deliberate ``{"email": null}`` working.
    """
    email: Optional[EmailStr]


class LlmEntitlementUpdate(BaseModel):
    """Body for the admin grant/revoke endpoint (issue #9)."""
    status: str  # "active" | "revoked"
    expires_at: Optional[datetime] = None
    notes: Optional[str] = None

    @field_validator("status")
    @classmethod
    def _valid_status(cls, v: str) -> str:
        if v not in ("active", "revoked"):
            raise ValueError('status must be "active" or "revoked"')
        return v


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
    # Send a provider-side strict JSON schema (``response_format``) for structured
    # generation. On by default; set to ``false`` for a server that doesn't
    # support it, to skip the wasted request + auto-fallback round-trip.
    structured_outputs: bool = True

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
    structured_outputs: bool = True


class InstanceSettingsResponse(BaseModel):
    llm_analysis_context: Optional[str]
    admin_contact: Optional[str]
    # The instance's entire LLM config: selectable presets, first = default.
    llm_models: list[LlmModelConfigOut] = []
    # Issue #9 opt-in gate: require an LLM-access entitlement (or BYOK).
    llm_requires_subscription: bool = False
    # Issue #15: allow self-serve email signup (also needs a configured provider).
    allow_self_signup: bool = False
    # Issue #46: allow users to issue personal access tokens. Defaults **on** —
    # unlike the gates above it preserves no prior behaviour, and a token grants
    # strictly less than the session its owner already holds. Turning it off
    # refuses authentication, so tokens already issued stop working at once.
    allow_personal_access_tokens: bool = True
    # Issue #42: expose the MCP tool server at POST /mcp. Defaults **on** for
    # the same reason as the switch above — it publishes read-only, scoped tools
    # over data the caller's credential already reaches. Off refuses the
    # endpoint outright, handshake included.
    allow_mcp_server: bool = True
    # Issue #56: offer course recon on this instance. Defaults **off**, unlike
    # the two above, because the half that distinguishes it — classifying the
    # road surface under a course — needs a routing sidecar the self-hoster
    # builds tiles for themselves. Off refuses the capability: every course and
    # bike endpoint, the background matcher and the plan generator. It never
    # refuses the data export, and it deletes nothing.
    allow_course_recon: bool = False


class InstanceSettingsPatch(BaseModel):
    llm_analysis_context: Optional[str] = None
    admin_contact: Optional[str] = None
    # Full-replacement list: send the complete desired preset list (first entry
    # is the default), or omit to leave unchanged.
    llm_models: Optional[list[LlmModelConfigIn]] = None
    llm_requires_subscription: Optional[bool] = None
    allow_self_signup: Optional[bool] = None
    allow_personal_access_tokens: Optional[bool] = None
    allow_mcp_server: Optional[bool] = None
    allow_course_recon: Optional[bool] = None


# ── LLM usage stats (instance admin, issue #9) ──────────────────────────────

class LlmUsageBucket(BaseModel):
    """One aggregation row of the usage summary.

    ``key`` is the group value (a user id, provider host, feature, or time
    bucket). Input and output tokens are summed **separately** — providers price
    them differently — and never merged into a single figure.
    """
    key: Optional[str] = None
    calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    unknown_usage_calls: int  # calls where the upstream omitted usage (nulls)


class LlmUsageSummaryResponse(BaseModel):
    group_by: str
    from_: Optional[str] = Field(default=None, serialization_alias="from")
    to: Optional[str] = None
    buckets: list[LlmUsageBucket] = []

    model_config = {"populate_by_name": True}


# ── Third-party API usage and quota headroom (issue #66) ────────────────────


class ApiUsageBucket(BaseModel):
    """One aggregation row of the third-party API-usage summary.

    ``calls`` counts **HTTP requests** for the providers and **messages** for
    email, which is the unit each is respectively limited and billed in. The
    outcome breakdown travels with the count because a bucket of 900 calls means
    something very different when 200 of them are ``rate_limited``.
    """
    key: Optional[str] = None
    calls: int
    ok: int
    client_error: int
    server_error: int
    transport_error: int
    rate_limited: int
    avg_duration_ms: Optional[int] = None


class ApiUsageSummaryResponse(BaseModel):
    group_by: str
    from_: Optional[str] = Field(default=None, serialization_alias="from")
    to: Optional[str] = None
    buckets: list[ApiUsageBucket] = []

    model_config = {"populate_by_name": True}


class QuotaWindow(BaseModel):
    """One quota window's standing for a service.

    ``observed_in_window`` is the field that stops this panel lying. When it is
    false the window has rolled over since we last called the provider, so
    ``usage`` is 0 by inference rather than by observation — the counter is
    genuinely empty, and showing last window's number instead would report
    alarming usage against a window nothing has spent.
    """
    window: str          # short | daily
    scope: str           # overall | read
    usage: Optional[int] = None
    limit: Optional[int] = None
    remaining: Optional[int] = None
    window_start: datetime
    resets_at: datetime
    observed_in_window: bool


class QuotaHeadroom(BaseModel):
    """A service's current headroom, with everything needed to read it honestly.

    ``observed_at``/``age_seconds`` are present because headroom is a *now*
    number that is only as fresh as our last call: a caller must be able to say
    "no observation in this window" rather than implying a live reading.
    """
    service: str
    observed_at: Optional[datetime] = None
    age_seconds: Optional[float] = None
    windows: list[QuotaWindow] = []
    last_rate_limited_at: Optional[datetime] = None


class QuotaHeadroomResponse(BaseModel):
    services: list[QuotaHeadroom] = []


class WebhookUsageBucket(BaseModel):
    """One aggregation row of the inbound-webhook summary.

    Counted at the bridge, which is the only place that sees a delivery
    *received*. The backend's own view would count redeliveries repeatedly and
    miss events shed by the queue ceiling entirely.
    """
    key: Optional[str] = None
    provider: Optional[str] = None
    outcome: Optional[str] = None
    count: int


class WebhookUsageSummaryResponse(BaseModel):
    group_by: str
    from_: Optional[str] = Field(default=None, serialization_alias="from")
    to: Optional[str] = None
    buckets: list[WebhookUsageBucket] = []
    #: Bridges that could not be reached for this request. The table is rendered
    #: from whatever did answer, with these named — a bridge being down is not a
    #: reason to show nothing, but it is a reason not to imply completeness.
    unavailable: list[str] = []

    model_config = {"populate_by_name": True}
