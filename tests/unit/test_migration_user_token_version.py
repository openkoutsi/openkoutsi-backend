"""Unit tests for registry migration 015 — users.token_version (issue #102, F-04).

Both the ORM model and the revision are required, and neither alone covers both
cases: ``create_all`` handles a fresh install, the revision handles an existing
volume. This exercises the second half, running the revision's own ``upgrade()``
against a real SQLite file through a real Alembic ``Operations`` proxy, so the
DDL itself is tested rather than a paraphrase of it.

The starting schema comes from the ORM metadata with this revision's addition
removed again, so the fixture cannot drift away from the database the migration
actually meets in production.

The load-bearing property here is the backfill: an instance upgrading with users
signed in must not sign them out, which means every existing row has to land on
0 — the same value a token minted before the column existed reports.
"""
import importlib
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text

from backend.app.db.base import RegistryBase
from backend.app.models.registry_orm import User

MIGRATION = "backend.app.db.migrations.registry.versions.015_user_token_version"


def _seed_pre_015(engine) -> None:
    """The registry as revision 014 left it: users, no token_version."""
    RegistryBase.metadata.create_all(engine, tables=[User.__table__])
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE users DROP COLUMN token_version"))
        for uid, name in (("u1", "existing"), ("u2", "also-existing")):
            conn.execute(text(
                "INSERT INTO users (id, username, password_hash, roles, created_at) "
                f"VALUES ('{uid}', '{name}', 'hash', '[\"user\"]', '2026-01-01 00:00:00')"
            ))


def _run(engine, direction: str = "upgrade") -> None:
    module = importlib.import_module(MIGRATION)
    with engine.begin() as conn:
        operations = Operations(MigrationContext.configure(conn))
        with patch.object(module, "op", operations):
            getattr(module, direction)()


@pytest.fixture
def migrated(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'registry.db'}")
    _seed_pre_015(engine)
    _run(engine)
    yield engine
    engine.dispose()


def test_it_follows_the_refresh_lock_revision():
    module = importlib.import_module(MIGRATION)
    assert module.revision == "015_user_token_version"
    assert module.down_revision == "014_provider_connection_refresh_lock"


def test_the_column_is_added(migrated):
    columns = {c["name"] for c in inspect(migrated).get_columns("users")}
    assert "token_version" in columns
    assert columns == {c.name for c in User.__table__.columns}


def test_existing_users_are_backfilled_to_zero(migrated):
    """An upgrade must not sign anyone out.

    0 is also what a token minted before this revision reports (no ``ver``
    claim), so those sessions stay valid until they expire — and the first
    reset after the upgrade moves the user to 1 and takes them with it.
    """
    with migrated.begin() as conn:
        versions = conn.execute(
            text("SELECT token_version FROM users ORDER BY id")
        ).scalars().all()
    assert versions == [0, 0]


def test_the_column_is_not_nullable(migrated):
    """A NULL would read as 'no generation' at exactly the wrong moment."""
    with pytest.raises(Exception):
        with migrated.begin() as conn:
            conn.execute(text("UPDATE users SET token_version = NULL WHERE id = 'u1'"))


def test_a_new_row_defaults_to_zero_without_naming_the_column(migrated):
    """Inserts written before this revision still work."""
    with migrated.begin() as conn:
        conn.execute(text(
            "INSERT INTO users (id, username, password_hash, roles, created_at) "
            "VALUES ('u3', 'fresh', 'hash', '[\"user\"]', '2026-01-01 00:00:00')"
        ))
        version = conn.execute(
            text("SELECT token_version FROM users WHERE id = 'u3'")
        ).scalar_one()
    assert version == 0


def test_downgrade_removes_the_column(migrated):
    _run(migrated, "downgrade")
    columns = {c["name"] for c in inspect(migrated).get_columns("users")}
    assert "token_version" not in columns
