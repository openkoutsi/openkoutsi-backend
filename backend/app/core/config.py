from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_INSECURE_DEFAULT = "changeme-set-a-real-secret-in-env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Root data directory — contains registry.db and teams/
    data_dir: str = "data"

    # Deprecated single-DB path kept for any transitional code during migration
    database_path: str = "openkoutsi.db"

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

    # LLM plan generation (OpenAI-compatible API)
    llm_base_url: str = ""   # e.g. "http://localhost:11434/v1" or "https://api.openai.com/v1"
    llm_api_key: str = ""    # empty is fine for local models
    llm_model: str = ""      # e.g. "llama3.2", "gpt-4o-mini", "mistral"

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

    # Field-level encryption key for sensitive DB columns (Fernet/base64-urlsafe, 32 bytes).
    # Generate: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    # Leave empty in development to disable encryption (tokens stored as plaintext).
    encryption_key: str = ""

    # Admin secret for privileged endpoints — replaced by JWT role checks in new arch,
    # kept for backward compatibility during transition.
    admin_secret: str | None = None

    # Secret for the superadmin panel (team approval). Set this in production.
    # Leave empty to disable the superadmin endpoints.
    superadmin_secret: str = ""

    # ── Path helpers ──────────────────────────────────────────────────────────

    @property
    def registry_db_path(self) -> str:
        return str(Path(self.data_dir) / "registry.db")

    def team_db_path(self, team_id: str) -> str:
        return str(Path(self.data_dir) / "teams" / team_id / "team.db")

    def user_data_dir(self, user_id: str) -> Path:
        return Path(self.data_dir) / "users" / user_id

    def user_db_path(self, user_id: str) -> str:
        # Generic name ("user.db") so other per-user data can live in this file.
        return str(self.user_data_dir(user_id) / "user.db")

    def team_fit_dir(self, team_id: str, global_user_id: str) -> Path:
        return Path(self.data_dir) / "teams" / team_id / "uploads" / global_user_id

    def team_avatar_dir(self, team_id: str) -> Path:
        return Path(self.data_dir) / "teams" / team_id / "avatars"


settings = Settings()
