"""Field-level encryption for sensitive database columns.

Registry DB columns (provider tokens) use the master ENCRYPTION_KEY directly.
Team DB columns use a per-team key set via set_team_encryption_context().

Generate a key:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""

from __future__ import annotations

import base64
from contextvars import ContextVar
from typing import Any

from sqlalchemy import String
from sqlalchemy.types import TypeDecorator

# Per-request context variable holding the current team's derived key bytes.
# Set by get_ctx_and_session before yielding to route handlers.
_team_key_var: ContextVar[bytes | None] = ContextVar("_team_key", default=None)


def set_team_encryption_context(team_id: str) -> None:
    """Derive and store the team key in the current async context."""
    from backend.app.core.config import settings
    if not settings.encryption_key:
        _team_key_var.set(None)
        return
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    raw_master = base64.urlsafe_b64decode(settings.encryption_key.encode())
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=f"team-key:{team_id}".encode(),
    )
    _team_key_var.set(hkdf.derive(raw_master))


def _get_registry_fernet():
    """Fernet using the raw master key — for registry DB columns."""
    from backend.app.core.config import settings
    if not settings.encryption_key:
        return None
    from cryptography.fernet import Fernet
    return Fernet(settings.encryption_key.encode())


def _get_team_fernet():
    """Fernet using the current team key — for team DB columns."""
    key_bytes = _team_key_var.get()
    if key_bytes is None:
        return None
    from cryptography.fernet import Fernet
    return Fernet(base64.urlsafe_b64encode(key_bytes))


class EncryptedString(TypeDecorator):
    """String column transparently encrypted/decrypted using the master key.

    Used in the registry DB for provider tokens.
    """

    impl = String
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        fernet = _get_registry_fernet()
        if fernet is None:
            return value
        return fernet.encrypt(value.encode()).decode()

    def process_result_value(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        fernet = _get_registry_fernet()
        if fernet is None:
            return value
        try:
            return fernet.decrypt(value.encode()).decode()
        except Exception:
            return value


class TeamEncryptedString(TypeDecorator):
    """String column encrypted with the current team's derived key.

    Used in team DB columns. Requires set_team_encryption_context() to be
    called before any DB operation within the request lifecycle.
    """

    impl = String
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        fernet = _get_team_fernet()
        if fernet is None:
            return value
        return fernet.encrypt(value.encode()).decode()

    def process_result_value(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        fernet = _get_team_fernet()
        if fernet is None:
            return value
        try:
            return fernet.decrypt(value.encode()).decode()
        except Exception:
            return value
