"""Unit tests for registry migration 012 — personal access tokens (issue #46).

Both the ORM model and the revision are required, and neither alone covers both
cases: ``create_all`` handles a fresh install, the revision handles an existing
volume. This exercises the second half, running the revision's own ``upgrade()``
against a real SQLite file through a real Alembic ``Operations`` proxy, so the
DDL itself is tested rather than a paraphrase of it.

The starting schema comes from the ORM metadata with this revision's additions
removed again, so the fixture cannot drift away from the database the migration
actually meets in production.
"""
import importlib
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text

from backend.app.db.base import RegistryBase
from backend.app.models.registry_orm import InstanceSettings, User

MIGRATION = "backend.app.db.migrations.registry.versions.012_personal_access_tokens"


def _seed_pre_012(engine) -> None:
    """The registry as revision 011 left it."""
    RegistryBase.metadata.create_all(
        engine, tables=[User.__table__, InstanceSettings.__table__]
    )
    with engine.begin() as conn:
        # Undo this revision's own additions to get back to the 011 schema.
        conn.execute(
            text("ALTER TABLE instance_settings DROP COLUMN allow_personal_access_tokens")
        )
        conn.execute(
            text(
                "INSERT INTO users (id, username, password_hash, roles, created_at) "
                "VALUES ('u1', 'existing', 'hash', '[\"user\"]', '2026-01-01 00:00:00')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO instance_settings (id, llm_requires_subscription, "
                "allow_self_signup, updated_at) VALUES (1, 0, 0, '2026-01-01 00:00:00')"
            )
        )


def _run(engine, direction: str = "upgrade") -> None:
    module = importlib.import_module(MIGRATION)
    with engine.begin() as conn:
        operations = Operations(MigrationContext.configure(conn))
        with patch.object(module, "op", operations):
            getattr(module, direction)()


@pytest.fixture
def migrated(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'registry.db'}")
    _seed_pre_012(engine)
    _run(engine)
    yield engine
    engine.dispose()


def test_it_follows_the_email_signup_revision():
    module = importlib.import_module(MIGRATION)
    assert module.revision == "012_personal_access_tokens"
    assert module.down_revision == "011_email_signup"


def test_the_table_is_created_with_every_column_the_model_declares(migrated):
    columns = {c["name"] for c in inspect(migrated).get_columns("personal_access_tokens")}
    from backend.app.models.registry_orm import PersonalAccessToken

    assert columns == {c.name for c in PersonalAccessToken.__table__.columns}


def test_the_token_hash_is_unique(migrated):
    with migrated.begin() as conn:
        conn.execute(text(
            "INSERT INTO personal_access_tokens "
            "(id, user_id, token_hash, name, scopes, expires_at, created_at) "
            "VALUES ('t1', 'u1', 'same', 'a', '[]', '2026-12-01', '2026-01-01')"
        ))
    with pytest.raises(Exception):
        with migrated.begin() as conn:
            conn.execute(text(
                "INSERT INTO personal_access_tokens "
                "(id, user_id, token_hash, name, scopes, expires_at, created_at) "
                "VALUES ('t2', 'u1', 'same', 'b', '[]', '2026-12-01', '2026-01-01')"
            ))


def test_user_id_is_indexed(migrated):
    """Verification is one indexed lookup; listing a user's tokens is another."""
    indexed = {
        tuple(ix["column_names"])
        for ix in inspect(migrated).get_indexes("personal_access_tokens")
    }
    assert ("user_id",) in indexed


def test_deleting_a_user_takes_their_tokens_with_them(migrated):
    with migrated.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        conn.execute(text(
            "INSERT INTO personal_access_tokens "
            "(id, user_id, token_hash, name, scopes, expires_at, created_at) "
            "VALUES ('t1', 'u1', 'h', 'a', '[]', '2026-12-01', '2026-01-01')"
        ))
        conn.execute(text("DELETE FROM users WHERE id = 'u1'"))
        remaining = conn.execute(
            text("SELECT COUNT(*) FROM personal_access_tokens")
        ).scalar_one()
    assert remaining == 0


def test_the_instance_toggle_defaults_on_for_an_existing_row(migrated):
    """An existing volume must not have the feature silently switched off — the
    admin was never told to go and turn it on."""
    with migrated.connect() as conn:
        value = conn.execute(
            text("SELECT allow_personal_access_tokens FROM instance_settings WHERE id = 1")
        ).scalar_one()
    assert value == 1


def test_the_toggle_is_not_null(migrated):
    column = next(
        c for c in inspect(migrated).get_columns("instance_settings")
        if c["name"] == "allow_personal_access_tokens"
    )
    assert column["nullable"] is False


def test_downgrade_removes_everything_it_added(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'registry.db'}")
    _seed_pre_012(engine)
    _run(engine, "upgrade")
    _run(engine, "downgrade")

    inspector = inspect(engine)
    assert "personal_access_tokens" not in inspector.get_table_names()
    assert "allow_personal_access_tokens" not in {
        c["name"] for c in inspector.get_columns("instance_settings")
    }
    engine.dispose()
