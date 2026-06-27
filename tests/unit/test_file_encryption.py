"""Unit tests for per-user FIT file encryption."""
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet

# A stable test key — generated once so tests are deterministic.
_TEST_KEY = Fernet.generate_key().decode()

_TEAM_ID = "test-team-enc"
_USER_ID = "user-abc"


def _patch_key(key=_TEST_KEY):
    from backend.app.core import config
    return patch.object(config.settings, "encryption_key", key)


class TestEncryptDecryptRoundtrip:
    def test_roundtrip_returns_original_bytes(self, tmp_path: Path):
        original = b"FIT\x0e\x10\x00\x00\x00.FIT" + b"\xff" * 100
        f = tmp_path / "test.fit"
        f.write_bytes(original)

        with _patch_key():
            from backend.app.core.file_encryption import decrypt_file, encrypt_file
            encrypt_file(f, _TEAM_ID, _USER_ID)
            result = decrypt_file(f, _TEAM_ID, _USER_ID)

        assert result == original

    def test_encrypted_file_does_not_match_original(self, tmp_path: Path):
        original = b"plaintext FIT content"
        f = tmp_path / "test.fit"
        f.write_bytes(original)

        with _patch_key():
            from backend.app.core.file_encryption import encrypt_file
            encrypt_file(f, _TEAM_ID, _USER_ID)

        assert f.read_bytes() != original

    def test_same_content_different_users_produces_different_ciphertext(self, tmp_path: Path):
        """Each user's derived key is independent — same plaintext encrypts differently."""
        data = b"identical content"
        fa = tmp_path / "a.fit"
        fb = tmp_path / "b.fit"
        fa.write_bytes(data)
        fb.write_bytes(data)

        with _patch_key():
            from backend.app.core.file_encryption import encrypt_file
            encrypt_file(fa, _TEAM_ID, "user-a")
            encrypt_file(fb, _TEAM_ID, "user-b")

        assert fa.read_bytes() != fb.read_bytes()

    def test_wrong_user_cannot_decrypt(self, tmp_path: Path):
        """Decrypting with a different user's key raises InvalidToken."""
        from cryptography.fernet import InvalidToken

        f = tmp_path / "test.fit"
        f.write_bytes(b"sensitive FIT data")

        with _patch_key():
            from backend.app.core.file_encryption import decrypt_file, encrypt_file
            encrypt_file(f, _TEAM_ID, "user-a")
            with pytest.raises(InvalidToken):
                decrypt_file(f, _TEAM_ID, "user-b")

    def test_missing_encryption_key_raises_runtime_error(self, tmp_path: Path):
        f = tmp_path / "test.fit"
        f.write_bytes(b"data")

        with _patch_key(key=None):
            from backend.app.core.file_encryption import encrypt_file
            with pytest.raises(RuntimeError, match="ENCRYPTION_KEY"):
                encrypt_file(f, _TEAM_ID, _USER_ID)

    def test_same_user_key_is_deterministic(self, tmp_path: Path):
        """Deriving the key twice for the same user yields the same Fernet instance."""
        from backend.app.core.file_encryption import _derive_fit_fernet

        with _patch_key():
            k1 = _derive_fit_fernet(_TEAM_ID, "user-xyz")
            k2 = _derive_fit_fernet(_TEAM_ID, "user-xyz")

        ciphertext = k1.encrypt(b"hello")
        assert k2.decrypt(ciphertext) == b"hello"


class TestEncryptDecryptSecret:
    """Tests for the small-secret (LLM API key) encrypt/decrypt helpers."""

    def test_roundtrip_returns_original_string(self):
        with _patch_key():
            from backend.app.core.file_encryption import decrypt_secret, encrypt_secret

            token = encrypt_secret("sk-mysecretkey", _TEAM_ID, _USER_ID)
            result = decrypt_secret(token, _TEAM_ID, _USER_ID)

        assert result == "sk-mysecretkey"

    def test_ciphertext_differs_from_plaintext(self):
        with _patch_key():
            from backend.app.core.file_encryption import encrypt_secret

            token = encrypt_secret("hunter2", _TEAM_ID, _USER_ID)

        assert "hunter2" not in token

    def test_different_users_produce_different_ciphertext(self):
        with _patch_key():
            from backend.app.core.file_encryption import encrypt_secret

            t1 = encrypt_secret("same-key", _TEAM_ID, "user-a")
            t2 = encrypt_secret("same-key", _TEAM_ID, "user-b")

        assert t1 != t2

    def test_wrong_user_cannot_decrypt(self):
        from cryptography.fernet import InvalidToken

        with _patch_key():
            from backend.app.core.file_encryption import decrypt_secret, encrypt_secret

            token = encrypt_secret("secret", _TEAM_ID, "user-a")
            with pytest.raises(InvalidToken):
                decrypt_secret(token, _TEAM_ID, "user-b")

    def test_secrets_key_is_independent_from_fit_file_key(self):
        """The LLM-key derivation uses a different HKDF info string."""
        with _patch_key():
            from backend.app.core.file_encryption import (
                _derive_fit_fernet,
                _derive_secret_fernet,
            )

            fit_key = _derive_fit_fernet(_TEAM_ID, "user-x")
            sec_key = _derive_secret_fernet(_TEAM_ID, "user-x")

        ciphertext = fit_key.encrypt(b"data")
        with pytest.raises(Exception):
            sec_key.decrypt(ciphertext)

    def test_missing_encryption_key_raises_runtime_error(self):
        with _patch_key(key=None):
            from backend.app.core.file_encryption import encrypt_secret

            with pytest.raises(RuntimeError, match="ENCRYPTION_KEY"):
                encrypt_secret("value", _TEAM_ID, _USER_ID)
