"""ENCRYPTION_KEY must be set, or plaintext must be chosen out loud (F-08).

``SECRET_KEY`` refuses to start on a weak or default value. ``ENCRYPTION_KEY``
had no equivalent: it defaulted to ``""`` and both column types read that as
"encryption off", so the Strava and Wahoo OAuth tokens on ``ProviderConnection``
went into the registry database as plaintext and nothing said so.

    # core/encryption.py
    fernet = _get_registry_fernet()
    if fernet is None:
        return value          # ← plaintext into the column

The inconsistency sat inside one subsystem: ``file_encryption`` raises outright
in the same situation, and it is what the LLM API-key helpers use. So the same
missing key was loud for FIT files and silent for provider tokens.

Settings now refuses to construct without a key unless ``ALLOW_PLAINTEXT_SECRETS``
says so, and the startup path says so again in the log when it is.
"""
import logging

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.core.config import Settings
from backend.app.db.base import RegistryBase
from backend.app.models.registry_orm import ProviderConnection

_STRONG_SECRET = "a" * 64
_VALID_KEY = Fernet.generate_key().decode()


def _settings(**overrides) -> Settings:
    """Build Settings from explicit values.

    Init arguments outrank the environment in pydantic-settings, so this is
    unaffected by whatever the suite exported for its own run.
    """
    return Settings(secret_key=_STRONG_SECRET, **overrides)


class TestAnEmptyKeyIsRefused:
    def test_missing_key_refuses_to_start(self):
        with pytest.raises(ValueError) as exc_info:
            _settings(encryption_key="", allow_plaintext_secrets=False)
        assert "ENCRYPTION_KEY" in str(exc_info.value)

    def test_the_message_explains_the_consequence_and_both_ways_out(self):
        """An operator hitting this at deploy time needs to act, not guess."""
        with pytest.raises(ValueError) as exc_info:
            _settings(encryption_key="", allow_plaintext_secrets=False)
        message = str(exc_info.value)
        assert "plaintext" in message
        assert "Fernet.generate_key" in message
        assert "ALLOW_PLAINTEXT_SECRETS" in message

    def test_default_is_to_refuse(self):
        """Not opting in is not the same as opting out."""
        with pytest.raises(ValueError):
            _settings(encryption_key="")


class TestPlaintextIsAnExplicitChoice:
    def test_opt_in_allows_an_empty_key(self):
        settings = _settings(encryption_key="", allow_plaintext_secrets=True)
        assert settings.encryption_key == ""

    def test_opt_in_does_not_weaken_secret_key(self):
        """The escape hatch is for one key, not for the validators generally."""
        with pytest.raises(ValueError) as exc_info:
            Settings(secret_key="short", encryption_key="", allow_plaintext_secrets=True)
        assert "SECRET_KEY" in str(exc_info.value)


class TestAnUnusableKeyIsRefused:
    @pytest.mark.parametrize("key", [
        "not-a-fernet-key",
        "c2hvcnQ=",                     # valid base64, wrong length
        _VALID_KEY[:-1],                # one character short
    ])
    def test_invalid_key_refuses_to_start(self, key):
        """Worse than no key: nothing fails until the first token is written."""
        with pytest.raises(ValueError) as exc_info:
            _settings(encryption_key=key)
        assert "ENCRYPTION_KEY" in str(exc_info.value)

    def test_a_valid_key_is_accepted(self):
        settings = _settings(encryption_key=_VALID_KEY)
        assert settings.encryption_key == _VALID_KEY


class TestTheColumnActuallyEncrypts:
    """What the silence was hiding, asserted directly against the database."""

    @pytest.fixture
    async def registry(self):
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(RegistryBase.metadata.create_all)
        yield engine, async_sessionmaker(engine, expire_on_commit=False)
        await engine.dispose()

    async def test_token_is_not_stored_as_plaintext(self, registry):
        """The suite runs with a real key, so this is the production shape."""
        engine, factory = registry
        secret = "strava-access-token-abc123"

        async with factory() as s:
            s.add(ProviderConnection(
                id="c1", user_id="u1", provider="strava", access_token=secret,
            ))
            await s.commit()

        # Read past the type decorator to see what is really on disk.
        async with engine.connect() as conn:
            stored = (await conn.execute(
                text("SELECT access_token FROM provider_connections WHERE id = 'c1'")
            )).scalar_one()

        assert stored != secret, "token was written to the column in plaintext"
        assert secret not in stored

    async def test_it_still_reads_back(self, registry):
        """Encrypted at rest is only useful if the round trip works."""
        _engine, factory = registry
        secret = "wahoo-refresh-token-xyz789"

        async with factory() as s:
            s.add(ProviderConnection(
                id="c2", user_id="u1", provider="wahoo", refresh_token=secret,
            ))
            await s.commit()

        async with factory() as s:
            loaded = await s.get(ProviderConnection, "c2")
            assert loaded.refresh_token == secret


class TestTheStartupWarning:
    async def test_plaintext_mode_is_logged_loudly(self, caplog):
        """The state has to be visible in a running instance's log.

        An operator who set ALLOW_PLAINTEXT_SECRETS months ago, or inherited an
        instance from someone who did, should not have to read the config to
        find out the tokens are unencrypted.
        """
        from unittest.mock import patch

        import backend.main as main_module

        with patch.object(main_module.settings, "encryption_key", ""):
            with caplog.at_level(logging.WARNING, logger="backend.main"):
                with patch.object(main_module, "init_registry_db") as init_reg:
                    init_reg.return_value = None
                    async with main_module.lifespan(object()):
                        pass

        assert "UNENCRYPTED" in caplog.text
        assert "ENCRYPTION_KEY" in caplog.text
