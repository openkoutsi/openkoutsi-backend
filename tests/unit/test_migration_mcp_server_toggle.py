"""Unit tests for registry migration 013 — the MCP server toggle (issue #42).

Same shape as ``test_migration_personal_access_tokens``, and for the same
reason: ``create_all`` covers a fresh install and the revision covers an existing
volume, and neither alone covers both. The starting schema is the ORM metadata
with this revision's own addition removed again, so the fixture cannot drift from
the database the migration actually meets in production.

The property that matters most here is the **default**. An instance upgrading
into this revision has an ``instance_settings`` row already, and if the new
column arrived null or false the MCP endpoint would silently be off on every
existing deployment — a feature that ships disabled for everyone who already runs
the software is not the feature that was reviewed.
"""
import importlib
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text

from backend.app.db.base import RegistryBase
from backend.app.models.registry_orm import InstanceSettings, User

MIGRATION = "backend.app.db.migrations.registry.versions.013_mcp_server_toggle"


def _seed_pre_013(engine) -> None:
    """The registry as revision 012 left it, with a settings row already in it."""
    RegistryBase.metadata.create_all(
        engine, tables=[User.__table__, InstanceSettings.__table__]
    )
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE instance_settings DROP COLUMN allow_mcp_server"))
        conn.execute(
            text(
                "INSERT INTO instance_settings (id, llm_requires_subscription, "
                "allow_self_signup, allow_personal_access_tokens, updated_at) "
                "VALUES (1, 0, 0, 1, '2026-01-01 00:00:00')"
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
    _seed_pre_013(engine)
    _run(engine)
    yield engine
    engine.dispose()


def test_it_follows_the_personal_access_token_revision():
    module = importlib.import_module(MIGRATION)
    assert module.revision == "013_mcp_server_toggle"
    assert module.down_revision == "012_personal_access_tokens"


def test_the_column_is_added(migrated):
    columns = {c["name"] for c in inspect(migrated).get_columns("instance_settings")}
    assert "allow_mcp_server" in columns


def test_an_existing_instance_comes_out_with_the_server_enabled(migrated):
    """The upgrade must not turn the feature off for everyone already running.

    The row predates the column, so the value it ends up with comes entirely
    from the server default.
    """
    with migrated.begin() as conn:
        value = conn.execute(
            text("SELECT allow_mcp_server FROM instance_settings WHERE id = 1")
        ).scalar_one()
    assert value == 1


def test_the_column_is_not_nullable(migrated):
    """A tri-state switch has a state nobody chose."""
    column = next(
        c for c in inspect(migrated).get_columns("instance_settings")
        if c["name"] == "allow_mcp_server"
    )
    assert column["nullable"] is False


def test_a_row_inserted_without_the_column_still_defaults_on(migrated):
    with migrated.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO instance_settings (id, llm_requires_subscription, "
                "allow_self_signup, allow_personal_access_tokens, updated_at) "
                "VALUES (2, 0, 0, 1, '2026-02-01 00:00:00')"
            )
        )
        value = conn.execute(
            text("SELECT allow_mcp_server FROM instance_settings WHERE id = 2")
        ).scalar_one()
    assert value == 1


def test_the_model_and_the_revision_agree(migrated):
    """``create_all`` builds a fresh install and the revision upgrades an old
    one; they have to land in the same place."""
    migrated_columns = {
        c["name"] for c in inspect(migrated).get_columns("instance_settings")
    }
    assert migrated_columns == {c.name for c in InstanceSettings.__table__.columns}


def test_the_downgrade_removes_it(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'registry.db'}")
    _seed_pre_013(engine)
    _run(engine)
    _run(engine, "downgrade")
    columns = {c["name"] for c in inspect(engine).get_columns("instance_settings")}
    assert "allow_mcp_server" not in columns
    engine.dispose()
