"""Unit tests for registry migration 016 — email-address changes (issue #62).

Both the ORM model and the revision are required, and neither alone covers both
cases: ``create_all`` handles a fresh install, the revision handles an existing
volume. This exercises the second half, running the revision's own ``upgrade()``
against a real SQLite file through a real Alembic ``Operations`` proxy, so the
DDL itself is tested rather than a paraphrase of it.

The starting schema comes from the ORM metadata with this revision's own table
left out, so the fixture cannot drift away from the database the migration
actually meets in production.
"""
import importlib
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from backend.app.db.base import RegistryBase
from backend.app.models.registry_orm import EmailChangeToken, User

MIGRATION = "backend.app.db.migrations.registry.versions.016_email_change"


def _seed_pre_016(engine) -> None:
    """The registry as revision 015 left it: users, and no change tokens."""
    RegistryBase.metadata.create_all(engine, tables=[User.__table__])
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO users (id, username, email, password_hash, roles, "
                "token_version, created_at) VALUES "
                "('u1', 'existing', 'u1@example.com', 'hash', '[\"user\"]', 0, "
                "'2026-01-01 00:00:00')"
            )
        )


def _run(engine, direction: str = "upgrade") -> None:
    module = importlib.import_module(MIGRATION)
    with engine.begin() as conn:
        operations = Operations(MigrationContext.configure(conn))
        with patch.object(module, "op", operations):
            getattr(module, direction)()


def _insert_token(conn, *, token_id: str, user_id: str, token_hash: str,
                  new_email: str = "new@example.com") -> None:
    conn.execute(
        text(
            "INSERT INTO email_change_tokens "
            "(id, user_id, token_hash, new_email, expires_at, created_at) VALUES "
            f"('{token_id}', '{user_id}', '{token_hash}', '{new_email}', "
            "'2030-01-01 00:00:00', '2026-01-01 00:00:00')"
        )
    )


@pytest.fixture
def migrated(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'registry.db'}")
    _seed_pre_016(engine)
    _run(engine)
    yield engine
    engine.dispose()


def test_it_follows_the_token_version_revision():
    module = importlib.import_module(MIGRATION)
    assert module.revision == "016_email_change"
    assert module.down_revision == "015_user_token_version"


def test_the_table_is_created_with_every_column_the_model_declares(migrated):
    columns = {c["name"] for c in inspect(migrated).get_columns("email_change_tokens")}
    assert columns == {c.name for c in EmailChangeToken.__table__.columns}


def test_new_email_is_required(migrated):
    """The address is the whole point of the row; a NULL one is a broken token."""
    with pytest.raises(IntegrityError):
        with migrated.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO email_change_tokens "
                    "(id, user_id, token_hash, expires_at, created_at) VALUES "
                    "('t1', 'u1', 'h1', '2030-01-01 00:00:00', '2026-01-01 00:00:00')"
                )
            )


def test_the_token_hash_is_unique(migrated):
    with migrated.begin() as conn:
        _insert_token(conn, token_id="t1", user_id="u1", token_hash="same")
    with pytest.raises(IntegrityError):
        with migrated.begin() as conn:
            _insert_token(conn, token_id="t2", user_id="u1", token_hash="same")


def test_the_same_address_may_be_pending_for_two_users(migrated):
    """Nothing is claimed until somebody confirms, so new_email is not unique."""
    with migrated.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO users (id, username, password_hash, roles, "
                "token_version, created_at) VALUES "
                "('u2', 'other', 'hash', '[\"user\"]', 0, '2026-01-01 00:00:00')"
            )
        )
        _insert_token(conn, token_id="t1", user_id="u1", token_hash="h1",
                      new_email="contested@example.com")
        _insert_token(conn, token_id="t2", user_id="u2", token_hash="h2",
                      new_email="contested@example.com")

    with migrated.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM email_change_tokens "
                 "WHERE new_email = 'contested@example.com'")
        ).scalar_one()
    assert count == 2


def test_deleting_a_user_takes_their_pending_changes_with_them(migrated):
    with migrated.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        _insert_token(conn, token_id="t1", user_id="u1", token_hash="h1")
        conn.execute(text("DELETE FROM users WHERE id = 'u1'"))
        remaining = conn.execute(
            text("SELECT COUNT(*) FROM email_change_tokens")
        ).scalar_one()
    assert remaining == 0


def test_downgrade_removes_everything_it_added(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'registry.db'}")
    _seed_pre_016(engine)
    _run(engine, "upgrade")
    assert "email_change_tokens" in inspect(engine).get_table_names()
    _run(engine, "downgrade")
    assert "email_change_tokens" not in inspect(engine).get_table_names()
    engine.dispose()
