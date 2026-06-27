"""Per-user file encryption for FIT files stored on disk.

Key hierarchy:
    master_key (ENCRYPTION_KEY env var)
        └─ HKDF(info="user-key:{user_id}") → user_key
               └─ HKDF(info="fit-file") → FIT file key
               └─ HKDF(info="user-secret") → user secret key

Each user's FIT files are encrypted with a key derived from the master key and
the user id, so a compromised user key cannot decrypt another user's files. No
per-user keys are stored; they are always re-derivable from the master key.

ENCRYPTION_KEY must be set. Unlike DB field encryption, file encryption raises
a hard error if the key is missing.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _derive_master_raw() -> bytes:
    from backend.app.core.config import settings
    if not settings.encryption_key:
        raise RuntimeError(
            "ENCRYPTION_KEY is not set — cannot encrypt/decrypt FIT files"
        )
    return base64.urlsafe_b64decode(settings.encryption_key.encode())


def _derive_user_key(user_id: str) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=f"user-key:{user_id}".encode(),
    )
    return hkdf.derive(_derive_master_raw())


def _derive_subkey_fernet(user_id: str, info: str):
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    user_key = _derive_user_key(user_id)
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=info.encode(),
    )
    derived = hkdf.derive(user_key)
    return Fernet(base64.urlsafe_b64encode(derived))


def encrypt_file(path: Path, user_id: str) -> None:
    """Encrypt the file at *path* in-place."""
    fernet = _derive_subkey_fernet(user_id, "fit-file")
    data = path.read_bytes()
    path.write_bytes(fernet.encrypt(data))
    log.debug("Encrypted %s (user=%s)", path, user_id)


def decrypt_file(path: Path, user_id: str) -> bytes:
    """Read and decrypt the file at *path*, returning plaintext bytes."""
    fernet = _derive_subkey_fernet(user_id, "fit-file")
    return fernet.decrypt(path.read_bytes())


# ── Small-secret encryption (user LLM API keys etc.) ──────────────────────

def encrypt_secret(plaintext: str, user_id: str) -> str:
    """Encrypt a short secret string. Returns a URL-safe Fernet token."""
    return _derive_subkey_fernet(user_id, "user-secret").encrypt(plaintext.encode()).decode()


def decrypt_secret(token: str, user_id: str) -> str:
    """Decrypt a Fernet token produced by encrypt_secret."""
    return _derive_subkey_fernet(user_id, "user-secret").decrypt(token.encode()).decode()


# ── Instance-level secret encryption (admin LLM API key) ──────────────────

def _derive_instance_secret_fernet():
    """Key for instance-level secrets (e.g. admin-configured LLM API key)."""
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"instance-secret",
    )
    derived = hkdf.derive(_derive_master_raw())
    return Fernet(base64.urlsafe_b64encode(derived))


def encrypt_instance_secret(plaintext: str) -> str:
    return _derive_instance_secret_fernet().encrypt(plaintext.encode()).decode()


def decrypt_instance_secret(token: str) -> str:
    return _derive_instance_secret_fernet().decrypt(token.encode()).decode()
