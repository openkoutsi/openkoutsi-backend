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

    @property
    def llm_allowed_servers_list(self) -> list[str]:
        if not self.llm_allowed_servers:
            return []
        return [s.strip() for s in self.llm_allowed_servers.split(",") if s.strip()]

    # Path to the dedicated LLM-usage database (append-only per-call token
    # accounting for instance-paid calls; issue #9). Kept in its own SQLite file
    # so its unbounded, high-volume rows can be pruned/rotated independently of
    # the registry DB. Leave empty to default to ``<data_dir>/llm_usage.db``.
    llm_usage_db: str = ""

    # Field-level encryption key for sensitive DB columns (Fernet/base64-urlsafe, 32 bytes).
    # Generate: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    # Leave empty in development to disable encryption (tokens stored as plaintext).
    encryption_key: str = ""

    # ── Privacy policy ─────────────────────────────────────────────────────────
    # openkoutsi is self-hosted: whoever runs an instance is the GDPR data
    # controller and must fill these in. They are rendered into the privacy
    # policy served at GET /api/legal/privacy-policy (see
    # backend/app/services/privacy_policy.py). Any field left blank keeps the
    # template's bracketed "[…]" placeholder, so an unconfigured instance renders
    # a clearly-incomplete policy rather than silently claiming false facts.
    privacy_app_name: str = "openkoutsi"
    privacy_policy_version: str = "1.0"          # kept in sync with consent version
    privacy_effective_date: str = ""             # e.g. "2026-07-14"
    privacy_last_updated: str = ""               # e.g. "2026-07-14"
    privacy_controller_name: str = ""
    privacy_controller_address: str = ""
    privacy_controller_country: str = ""
    privacy_controller_registration_number: str = ""
    privacy_contact_email: str = ""
    privacy_dpo: str = ""                         # Data Protection Officer, if any
    privacy_eu_representative: str = ""           # Art. 27 representative, if any
    privacy_hosting_provider: str = ""           # e.g. "Hetzner, Finland (EU)"
    privacy_encryption_description: str = ""      # how data is encrypted at rest
    privacy_retention_period: str = ""            # after account deletion
    privacy_diagnostic_retention_period: str = ""
    privacy_children_min_age: int = 16            # age of digital consent
    privacy_supervisory_authority: str = ""       # lead DPA + complaint page

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
