"""A value that will not decrypt is not the value (issue #102, F-12).

``process_result_value`` caught *every* exception from ``fernet.decrypt`` and
returned the raw column:

    try:
        return fernet.decrypt(value.encode()).decode()
    except Exception:
        return value

That was presumably there to tolerate rows written before encryption was
enabled. But it makes a wrong key, a rotated key and a corrupted row
indistinguishable from a plaintext row — and the ciphertext then travels on as
though it were the value, reaching the provider as a bearer credential and
failing there with an error about the wrong thing.

The two cases collapse into one ``InvalidToken`` from ``decrypt``, so they are
told apart by *structure* before decrypting: a Fernet token has a recognisable
shape, and anything that does not have it was never ciphertext to begin with.
"""
import base64
import logging
import uuid

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.core import encryption
from backend.app.core.encryption import _decrypt_column, _looks_encrypted
from backend.app.db.base import RegistryBase
from backend.app.models.registry_orm import ProviderConnection

_OTHER_KEY = Fernet.generate_key()


class TestLooksEncrypted:
    def test_a_real_token_is_recognised(self):
        token = Fernet(Fernet.generate_key()).encrypt(b"strava-token").decode()
        assert _looks_encrypted(token)

    @pytest.mark.parametrize("value", [
        "",
        "plain-token",
        "a1b2c3d4e5f6",
        "strava-access-token-abc123",
        "not base64 at all !!!",
        "üüüü",                        # non-ASCII
        base64.urlsafe_b64encode(b"short").decode(),   # valid base64, too short
    ])
    def test_plaintext_is_not_mistaken_for_ciphertext(self, value):
        assert not _looks_encrypted(value)

    def test_a_long_base64_string_with_the_wrong_version_byte_is_not(self):
        """Length alone is not enough — the version byte is what pins it."""
        raw = bytes([0x79]) + b"x" * 80
        assert not _looks_encrypted(base64.urlsafe_b64encode(raw).decode())


class TestDecryptColumn:
    def test_a_readable_token_round_trips(self):
        key = Fernet.generate_key()
        fernet = Fernet(key)
        token = fernet.encrypt(b"secret-value").decode()
        assert _decrypt_column(fernet, token, column="c") == "secret-value"

    def test_legacy_plaintext_passes_through(self):
        """The one case the old blanket `except` was actually there to serve."""
        fernet = Fernet(Fernet.generate_key())
        assert _decrypt_column(fernet, "written-before-encryption", column="c") == (
            "written-before-encryption"
        )

    def test_a_token_from_another_key_returns_none(self, caplog):
        written_with = Fernet(_OTHER_KEY)
        read_with = Fernet(Fernet.generate_key())
        token = written_with.encrypt(b"secret-value").decode()

        with caplog.at_level(logging.ERROR, logger="backend.app.core.encryption"):
            result = _decrypt_column(read_with, token, column="c")

        assert result is None
        assert "ENCRYPTION_KEY" in caplog.text

    def test_the_ciphertext_is_never_returned(self):
        """The finding, stated directly."""
        written_with = Fernet(_OTHER_KEY)
        read_with = Fernet(Fernet.generate_key())
        token = written_with.encrypt(b"secret-value").decode()

        assert _decrypt_column(read_with, token, column="c") != token

    def test_a_corrupted_token_returns_none(self):
        fernet = Fernet(Fernet.generate_key())
        token = fernet.encrypt(b"secret-value").decode()
        corrupted = token[:-4] + "AAAA"

        assert _decrypt_column(fernet, corrupted, column="c") is None

    def test_unexpected_errors_are_not_swallowed(self):
        """`except Exception` hid programming errors too, not just bad tokens."""
        class _Exploding:
            def decrypt(self, _value):
                raise RuntimeError("something else entirely")

        token = Fernet(Fernet.generate_key()).encrypt(b"v").decode()
        with pytest.raises(RuntimeError, match="something else entirely"):
            _decrypt_column(_Exploding(), token, column="c")


class TestThroughTheColumn:
    """End to end on ProviderConnection, where the tokens actually live."""

    @pytest.fixture
    async def registry(self):
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(RegistryBase.metadata.create_all)
        yield engine, async_sessionmaker(engine, expire_on_commit=False)
        await engine.dispose()

    async def _insert_raw(self, engine, conn_id: str, stored: str) -> None:
        """Write straight past the type decorator, as an older row would be."""
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO provider_connections "
                    "(id, user_id, provider, access_token, created_at, updated_at) "
                    "VALUES (:id, 'u1', 'strava', :tok, "
                    "'2026-01-01 00:00:00', '2026-01-01 00:00:00')"
                ),
                {"id": conn_id, "tok": stored},
            )

    async def test_a_normal_token_round_trips(self, registry):
        _engine, factory = registry
        async with factory() as s:
            s.add(ProviderConnection(
                id="c1", user_id="u1", provider="strava", access_token="live-token"
            ))
            await s.commit()

        async with factory() as s:
            assert (await s.get(ProviderConnection, "c1")).access_token == "live-token"

    async def test_a_pre_encryption_row_still_reads(self, registry):
        """Upgrading an instance that ran without a key must not break it."""
        engine, factory = registry
        await self._insert_raw(engine, "c2", "plaintext-legacy-token")

        async with factory() as s:
            assert (await s.get(ProviderConnection, "c2")).access_token == (
                "plaintext-legacy-token"
            )

    async def test_a_row_from_another_key_reads_as_none(self, registry, caplog):
        engine, factory = registry
        foreign = Fernet(_OTHER_KEY).encrypt(b"someone-elses-token").decode()
        await self._insert_raw(engine, "c3", foreign)

        with caplog.at_level(logging.ERROR, logger="backend.app.core.encryption"):
            async with factory() as s:
                loaded = await s.get(ProviderConnection, "c3")

        assert loaded.access_token is None
        assert loaded.access_token != foreign, "ciphertext came back as the token"
        assert "could not decrypt" in caplog.text.lower()

    async def test_one_unreadable_row_does_not_break_the_query(self, registry):
        """Why None rather than raising.

        Raising in `process_result_value` aborts the whole result set, so a
        single bad row would take down every query touching the table —
        including the ones needed to diagnose it.
        """
        engine, factory = registry
        await self._insert_raw(
            engine, "c4", Fernet(_OTHER_KEY).encrypt(b"bad").decode()
        )
        async with factory() as s:
            s.add(ProviderConnection(
                id="c5", user_id="u2", provider="wahoo", access_token="good-token"
            ))
            await s.commit()

        async with factory() as s:
            from sqlalchemy import select

            rows = (await s.execute(
                select(ProviderConnection).order_by(ProviderConnection.id)
            )).scalars().all()

        assert [r.access_token for r in rows] == [None, "good-token"]
