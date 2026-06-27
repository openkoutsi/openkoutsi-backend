"""Per-team, per-user file encryption for FIT files stored on disk.

Key hierarchy:
    master_key (ENCRYPTION_KEY env var)
        └─ HKDF(info="team-key:{team_id}") → team_key
               └─ HKDF(info="fit-file:{global_user_id}") → FIT file key

Each user's FIT files are encrypted with a key derived from both the team and
the user, so:
- A compromised team key cannot decrypt another team's files.
- A compromised user key cannot decrypt another user's files.
- No per-user or per-team keys are stored; they are always re-derivable from
  the master key.

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


def _derive_team_key(team_id: str) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=f"team-key:{team_id}".encode(),
    )
    return hkdf.derive(_derive_master_raw())


def _derive_fit_fernet(team_id: str, global_user_id: str):
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    team_key = _derive_team_key(team_id)
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=f"fit-file:{global_user_id}".encode(),
    )
    derived = hkdf.derive(team_key)
    return Fernet(base64.urlsafe_b64encode(derived))


def encrypt_file(path: Path, team_id: str, global_user_id: str) -> None:
    """Encrypt the file at *path* in-place."""
    fernet = _derive_fit_fernet(team_id, global_user_id)
    data = path.read_bytes()
    path.write_bytes(fernet.encrypt(data))
    log.debug("Encrypted %s (team=%s user=%s)", path, team_id, global_user_id)


def decrypt_file(path: Path, team_id: str, global_user_id: str) -> bytes:
    """Read and decrypt the file at *path*, returning plaintext bytes."""
    fernet = _derive_fit_fernet(team_id, global_user_id)
    return fernet.decrypt(path.read_bytes())


# ── Small-secret encryption (LLM API keys etc.) ───────────────────────────

def _derive_secret_fernet(team_id: str, global_user_id: str):
    """Distinct key for user secrets — independent of the FIT key."""
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    team_key = _derive_team_key(team_id)
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=f"user-secret:{global_user_id}".encode(),
    )
    derived = hkdf.derive(team_key)
    return Fernet(base64.urlsafe_b64encode(derived))


def encrypt_secret(plaintext: str, team_id: str, global_user_id: str) -> str:
    """Encrypt a short secret string. Returns a URL-safe Fernet token."""
    fernet = _derive_secret_fernet(team_id, global_user_id)
    return fernet.encrypt(plaintext.encode()).decode()


def decrypt_secret(token: str, team_id: str, global_user_id: str) -> str:
    """Decrypt a Fernet token produced by encrypt_secret."""
    fernet = _derive_secret_fernet(team_id, global_user_id)
    return fernet.decrypt(token.encode()).decode()


# ── Team-level secret encryption (admin LLM API key) ─────────────────────

def _derive_team_secret_fernet(team_id: str):
    """Key for team-level secrets (e.g. admin-configured LLM API key)."""
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=f"team-secret:{team_id}".encode(),
    )
    derived = hkdf.derive(_derive_master_raw())
    return Fernet(base64.urlsafe_b64encode(derived))


def encrypt_team_secret(plaintext: str, team_id: str) -> str:
    return _derive_team_secret_fernet(team_id).encrypt(plaintext.encode()).decode()


def decrypt_team_secret(token: str, team_id: str) -> str:
    return _derive_team_secret_fernet(team_id).decrypt(token.encode()).decode()
