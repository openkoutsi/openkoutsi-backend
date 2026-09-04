"""Field-level encryption for sensitive database columns.

Registry DB columns (provider tokens) use the master ENCRYPTION_KEY directly.
Per-user DB columns use a per-user key set via set_user_encryption_context().

Generate a key:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""

from __future__ import annotations

import base64
import logging
from contextvars import ContextVar
from typing import Any

from sqlalchemy import String
from sqlalchemy.types import TypeDecorator

# Per-request context variable holding the current user's derived key bytes.
# Set by get_ctx_and_session before yielding to route handlers.
log = logging.getLogger(__name__)

_user_key_var: ContextVar[bytes | None] = ContextVar("_user_key", default=None)


def set_user_encryption_context(user_id: str) -> None:
    """Derive and store the user key in the current async context."""
    from backend.app.core.config import settings
    if not settings.encryption_key:
        _user_key_var.set(None)
        return
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    raw_master = base64.urlsafe_b64decode(settings.encryption_key.encode())
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=f"user-key:{user_id}".encode(),
    )
    _user_key_var.set(hkdf.derive(raw_master))


def _get_registry_fernet():
    """Fernet using the raw master key — for registry DB columns."""
    from backend.app.core.config import settings
    if not settings.encryption_key:
        return None
    from cryptography.fernet import Fernet
    return Fernet(settings.encryption_key.encode())


def _get_user_fernet():
    """Fernet using the current user key — for per-user DB columns."""
    key_bytes = _user_key_var.get()
    if key_bytes is None:
        return None
    from cryptography.fernet import Fernet
    return Fernet(base64.urlsafe_b64encode(key_bytes))


# A Fernet token is base64url over: version byte 0x80, an 8-byte timestamp, a
# 16-byte IV, the ciphertext, and a 32-byte HMAC. 73 bytes is the shortest one
# that can exist (16 bytes of ciphertext being the minimum AES block).
_FERNET_VERSION = 0x80
_FERNET_MIN_BYTES = 73


def _looks_encrypted(value: str) -> bool:
    """Whether *value* is structurally a Fernet token.

    `fernet.decrypt` raises ``InvalidToken`` for both of the cases that have to
    be told apart here: a row written before encryption was enabled, which is
    plaintext and must pass through, and a row that *is* ciphertext this key
    cannot open — a wrong key, a rotated key, a corrupted value — which must
    not be handed back as though it were the value (issue #102, F-12).

    Structure is what separates them, so it is checked before decrypting rather
    than inferred from the failure.
    """
    try:
        raw = base64.urlsafe_b64decode(value.encode("ascii"))
    except (UnicodeEncodeError, ValueError):
        # Not base64 at all, or not ASCII: cannot be something we wrote.
        return False
    return len(raw) >= _FERNET_MIN_BYTES and raw[0] == _FERNET_VERSION


def _decrypt_column(fernet, value: str, *, column: str) -> str | None:
    """Decrypt one column value, or report that it cannot be read.

    Returns ``None`` rather than raising when a genuine ciphertext will not open:
    raising aborts the whole result set, so one unreadable row would take down
    every query touching the table, including the ones needed to diagnose it.
    ``None`` is a value these columns already hold and callers already handle,
    and the provider write-back only happens after a *successful* call.

    What it must never do is return the ciphertext — which came back looking like
    a live token, went to the provider as a bearer credential, and failed there
    with an error about the wrong thing.
    """
    from cryptography.fernet import InvalidToken

    if not _looks_encrypted(value):
        # Written before encryption was enabled. Passing it through unchanged
        # is the migration path, and is the only case the old blanket `except`
        # was actually there to serve.
        return value

    try:
        return fernet.decrypt(value.encode()).decode()
    except InvalidToken:
        log.error(
            "Could not decrypt %s: the stored value is a Fernet token this key "
            "cannot open. Usually ENCRYPTION_KEY has changed since the value "
            "was written; it can also mean a corrupted row. Returning None, so "
            "this reads as 'no value' rather than as the ciphertext. Restore "
            "the original key to recover it.",
            column,
        )
        return None


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
        return _decrypt_column(fernet, value, column="a registry column")


class UserEncryptedString(TypeDecorator):
    """String column encrypted with the current user's derived key.

    Used in per-user DB columns. Requires set_user_encryption_context() to be
    called before any DB operation within the request lifecycle.
    """

    impl = String
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        fernet = _get_user_fernet()
        if fernet is None:
            return value
        return fernet.encrypt(value.encode()).decode()

    def process_result_value(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        fernet = _get_user_fernet()
        if fernet is None:
            return value
        return _decrypt_column(fernet, value, column="a per-user column")
