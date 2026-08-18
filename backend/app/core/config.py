from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_INSECURE_DEFAULT = "changeme-set-a-real-secret-in-env"


class Settings(BaseSettings):
    # In containers, secret fields are delivered as files under /run/secrets/
    # (Docker secrets) and read by pydantic-settings; non-secret config stays in
    # the environment. For local dev the .env workflow keeps working — env vars
    # take precedence over file secrets, so set only one source per field.
    model_config = SettingsConfigDict(
        env_file=".env", secrets_dir="/run/secrets", extra="ignore"
    )

    # Root data directory — contains registry.db and users/
    data_dir: str = "data"

    secret_key: str = _INSECURE_DEFAULT

    @model_validator(mode="after")
    def _validate_secret_key(self) -> "Settings":
        if self.secret_key == _INSECURE_DEFAULT or len(self.secret_key) < 32:
            raise ValueError(
                "SECRET_KEY is not set or is too weak. "
                "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        return self
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 30
    file_storage_path: str = "uploads"
    frontend_url: str = "http://localhost:3000"
    api_url: str = "http://localhost:8000"

    # Strava
    strava_client_id: str = ""
    strava_client_secret: str = ""
    bridge_url: str = ""
    bridge_secret: str = ""

    # Wahoo (register at developers.wahooligan.com)
    wahoo_client_id: str = ""
    wahoo_client_secret: str = ""
    wahoo_bridge_url: str = ""
    wahoo_bridge_secret: str = ""

    # Comma-separated list of LLM base URLs that users are allowed to choose from.
    # When set, users can only pick from this list; the free-text URL input is hidden.
    # When empty (default), users may enter any URL (subject to SSRF guards).
    # Example: "http://localhost:11434/v1,https://api.openai.com/v1"
    llm_allowed_servers: str = ""

    # Allow LLM base URLs that resolve to loopback, RFC 1918, ULA or CGNAT
    # addresses. Off by default: the base URL is user-supplied, so leaving the
    # private address space reachable makes it a probe against whatever else
    # runs on this host and its network. Turn it on for a self-hosted model
    # (Ollama on localhost, a model server on the LAN) — it does not re-open
    # the cloud metadata ranges, which stay blocked either way.
    llm_allow_private_networks: bool = False

    @property
    def llm_allowed_servers_list(self) -> list[str]:
        if not self.llm_allowed_servers:
            return []
        return [s.strip() for s in self.llm_allowed_servers.split(",") if s.strip()]

    # How many agentic coaching runs (issue #43) may be in flight at once in this
    # process. An agent loop is three to five completions instead of one, and a
    # local model that serialises requests turns a handful of concurrent runs
    # into a queue nobody is watching. A run that cannot get a slot immediately
    # falls back to the single-shot blob prompt rather than waiting — degrading
    # to the cheaper answer beats sitting on a spinner until the 30-minute
    # pending timeout. Optional; the default suits an instance with a hosted
    # provider. Lower it (1–2) for a single local GPU.
    agent_max_concurrent_runs: int = 4

    # ── Conversational Koutsi (issue #44) ─────────────────────────────────────
    # Chat is the first LLM surface the *athlete* can trigger arbitrarily often,
    # and every turn is a full agent run rather than one completion. Everything
    # else in the platform is bounded by "one ride, one analysis" or "once a
    # day"; these are the bounds that replace that.

    # How long an interactive turn may sit queued waiting for one of the
    # `agent_max_concurrent_runs` slots before giving up. Unlike a background
    # run, chat has no single-shot prompt to degrade to, so refusing instantly
    # would just lose the athlete's question — but the wait has to be bounded,
    # or it becomes the spinner the immediate refusal was designed to avoid.
    chat_queue_wait_seconds: float = 45.0

    # Tool-calling rounds a chat turn may spend before it is made to answer.
    # Lower than the status card's six: a card is one broad question wanting
    # several lookups, while a conversation can ask its follow-up as a *turn*
    # instead of spending a round on it.
    chat_max_rounds: int = 4

    # Turns per rolling day, and per single conversation. The per-conversation
    # cap is not primarily about cost — it is the point past which replaying a
    # thread is worse value than starting a fresh one.
    chat_max_turns_per_day: int = 50
    chat_max_turns_per_conversation: int = 40

    # Longest single question accepted, in characters. Also what stops one
    # pathological message from consuming the whole history budget below.
    chat_max_message_chars: int = 4000

    # Character budget for replayed dialogue on each turn, before the system
    # prompt and this turn's tool results. Tool results are never stored
    # (`models/chat_orm`), so this bounds prose only and can be generous.
    chat_history_chars: int = 12000

    # Minutes without a progress commit before a chat turn is declared dead.
    # Shorter than the daily card's 30 — that runs in the background with nobody
    # watching, and this has someone waiting on it — but not as short as it first
    # looks like it could be. The clock is touched by progress markers and text
    # flushes, and a tool round emits one marker and then no text at all while
    # the model composes the call and reasons over the result, so the gap between
    # two commits is a whole completion on a slow local model. Three minutes
    # declared healthy runs dead; ten is still bounded, and an athlete who has
    # waited that long has navigated away and will have the row resume on return.
    chat_stuck_minutes: int = 10

    # Path to the dedicated LLM-usage database (append-only per-call token
    # accounting for instance-paid calls; issue #9). Kept in its own SQLite file
    # so its unbounded, high-volume rows can be pruned/rotated independently of
    # the registry DB. Leave empty to default to ``<data_dir>/llm_usage.db``.
    llm_usage_db: str = ""

    # Field-level encryption key for sensitive DB columns (Fernet/base64-urlsafe, 32 bytes).
    # Generate: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    # Required unless `allow_plaintext_secrets` is set — see the validator below.
    encryption_key: str = ""

    # Run without ENCRYPTION_KEY, storing provider OAuth tokens in the registry
    # database as plaintext. An explicit, logged choice rather than the default:
    # an empty key used to disable encryption silently, so an instance that
    # never set one looked exactly like an instance that had (issue #102, F-08).
    # Development and throwaway instances are what this is for.
    allow_plaintext_secrets: bool = False

    @model_validator(mode="after")
    def _validate_encryption_key(self) -> "Settings":
        if not self.encryption_key:
            if self.allow_plaintext_secrets:
                return self
            raise ValueError(
                "ENCRYPTION_KEY is not set, so Strava and Wahoo OAuth tokens "
                "would be stored as plaintext in the registry database. "
                "Generate one with: python -c \"from cryptography.fernet import "
                "Fernet; print(Fernet.generate_key().decode())\" — or set "
                "ALLOW_PLAINTEXT_SECRETS=true to accept plaintext deliberately."
            )

        # A key that Fernet cannot load is worse than no key: nothing complains
        # until the first token is written, and then every provider connection
        # fails at runtime. Same reasoning as SECRET_KEY above — an instance
        # that cannot do the job should not come up claiming it can.
        from cryptography.fernet import Fernet

        try:
            Fernet(self.encryption_key.encode())
        except Exception as exc:
            raise ValueError(
                f"ENCRYPTION_KEY is not a valid Fernet key ({exc}). Generate one "
                'with: python -c "from cryptography.fernet import Fernet; '
                'print(Fernet.generate_key().decode())"'
            )
        return self

    # ── Email (transactional + inbound) ───────────────────────────────────────
    # Everything provider-specific lives behind the shared email module
    # (backend/app/services/email/); these settings only select and configure a
    # provider. All are optional — when unset, outbound email degrades gracefully
    # (callers should check before offering email-dependent features) and there
    # is no inbound surface.

    # Which EmailProvider implementation to use. "lettermint" and "euromail" are
    # available today; swapping providers should touch only the email module.
    email_provider: str = "lettermint"

    # Sender address for outbound transactional mail (e.g. verification and
    # password-reset messages). Required to actually send.
    email_from: str = ""

    # Lettermint (https://lettermint.co) — EU-based transactional email provider.
    # API token for outbound sends (delivered as a Docker secret in production).
    lettermint_api_key: str = ""

    # Signing secret for verifying inbound Lettermint webhooks. Used by the
    # optional inbound-email bridge (issue #38) to authenticate the provider's
    # POSTs before they reach the backend.
    lettermint_webhook_secret: str = ""

    # EuroMail (https://euromail.dev) — EU-based (Finland) transactional email
    # provider whose free tier includes inbound email (issue #41). API token for
    # outbound sends (delivered as a Docker secret in production).
    euromail_api_key: str = ""

    # Signing secret for verifying inbound EuroMail webhooks (HMAC-SHA256 over the
    # X-Euromail-Signature header). Used by the optional inbound-email bridge.
    euromail_webhook_secret: str = ""

    # URL of the privacy policy shown on the consent screen and auth pages.
    # Defaults to the canonical koutsi.dev policy; self-hosters are their own GDPR
    # data controller and should point this at their own policy. Exposed to the
    # frontend via GET /api/public/instance-info.
    privacy_policy_url: str = "https://koutsi.dev/privacy"

    # ── Path helpers ──────────────────────────────────────────────────────────

    @property
    def registry_db_path(self) -> str:
        return str(Path(self.data_dir) / "registry.db")

    @property
    def llm_usage_db_path(self) -> str:
        """Filesystem path of the dedicated LLM-usage database.

        Configurable via ``LLM_USAGE_DB``; defaults to ``<data_dir>/llm_usage.db``.
        """
        if self.llm_usage_db:
            return self.llm_usage_db
        return str(Path(self.data_dir) / "llm_usage.db")

    def user_data_dir(self, user_id: str) -> Path:
        return Path(self.data_dir) / "users" / user_id

    def user_db_path(self, user_id: str) -> str:
        # Generic name ("user.db") so all per-user data lives in this one file.
        return str(self.user_data_dir(user_id) / "user.db")

    def user_fit_dir(self, user_id: str) -> Path:
        return self.user_data_dir(user_id) / "uploads"

    def user_avatar_dir(self, user_id: str) -> Path:
        return self.user_data_dir(user_id) / "avatars"


settings = Settings()
