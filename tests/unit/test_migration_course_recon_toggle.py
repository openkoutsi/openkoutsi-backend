"""Unit tests for registry migration 018 — the course recon toggle (issue #56).

Same shape as ``test_migration_mcp_server_toggle``, and for the same reason:
``create_all`` covers a fresh install and the revision covers an existing
volume, and neither alone covers both. The starting schema is the ORM metadata
with this revision's own addition removed again, so the fixture cannot drift
from the database the migration actually meets in production.

The property that matters most here is the **default**, and it is the opposite
of 013's. The MCP toggle had to arrive *on*, because shipping a reviewed
feature disabled for every existing deployment is not the feature that was
reviewed. This one has to arrive *off*, because the half of course recon that
distinguishes it needs a routing sidecar with tiles the self-hoster builds
themselves — and an instance that has made no decision about that has not
implicitly said yes.

That difference is visible on upgrade, so it is asserted here rather than left
to be discovered: an instance already using course recon comes out of this
migration with it switched off, and nothing it stored is touched.
"""
import importlib
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text

from backend.app.db.base import RegistryBase
from backend.app.models.registry_orm import InstanceSettings, User

MIGRATION = "backend.app.db.migrations.registry.versions.018_course_recon_toggle"


def _seed_pre_018(engine) -> None:
    """The registry as revision 017 left it, with a settings row already in it."""
    RegistryBase.metadata.create_all(
        engine, tables=[User.__table__, InstanceSettings.__table__]
    )
    with engine.begin() as conn:
        conn.execute(
            text("ALTER TABLE instance_settings DROP COLUMN allow_course_recon")
        )
        conn.execute(
            text(
                "INSERT INTO instance_settings (id, llm_requires_subscription, "
                "allow_self_signup, allow_personal_access_tokens, allow_mcp_server, "
                "updated_at) VALUES (1, 0, 0, 1, 1, '2026-01-01 00:00:00')"
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
    _seed_pre_018(engine)
    _run(engine)
    yield engine
    engine.dispose()


def test_it_follows_the_registry_leases_revision():
    module = importlib.import_module(MIGRATION)
    assert module.revision == "018_course_recon_toggle"
    assert module.down_revision == "017_registry_leases"


def test_the_column_is_added(migrated):
    columns = {c["name"] for c in inspect(migrated).get_columns("instance_settings")}
    assert "allow_course_recon" in columns


def test_an_existing_instance_comes_out_with_course_recon_off(migrated):
    """The deliberate regression, asserted rather than discovered.

    Course recon shipped ungated, so an instance already using it loses course
    upload until an admin turns this on. That is the decision the issue asks
    for — the capability needs infrastructure nobody has consented to yet — but
    it is visible, so it belongs in a test and in the deployment guide rather
    than in a surprise.
    """
    with migrated.begin() as conn:
        value = conn.execute(
            text("SELECT allow_course_recon FROM instance_settings WHERE id = 1")
        ).scalar_one()
    assert value == 0


def test_the_column_is_not_nullable(migrated):
    """A tri-state switch has a state nobody chose."""
    column = next(
        c
        for c in inspect(migrated).get_columns("instance_settings")
        if c["name"] == "allow_course_recon"
    )
    assert column["nullable"] is False


def test_a_row_inserted_without_the_column_still_defaults_off(migrated):
    with migrated.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO instance_settings (id, llm_requires_subscription, "
                "allow_self_signup, allow_personal_access_tokens, allow_mcp_server, "
                "updated_at) VALUES (2, 0, 0, 1, 1, '2026-02-01 00:00:00')"
            )
        )
        value = conn.execute(
            text("SELECT allow_course_recon FROM instance_settings WHERE id = 2")
        ).scalar_one()
    assert value == 0


def test_the_model_and_the_revision_agree(migrated):
    """``create_all`` builds a fresh install and the revision upgrades an old
    one; they have to land in the same place."""
    migrated_columns = {
        c["name"] for c in inspect(migrated).get_columns("instance_settings")
    }
    assert migrated_columns == {c.name for c in InstanceSettings.__table__.columns}


def test_the_downgrade_removes_it(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'registry.db'}")
    _seed_pre_018(engine)
    _run(engine)
    _run(engine, "downgrade")
    columns = {c["name"] for c in inspect(engine).get_columns("instance_settings")}
    assert "allow_course_recon" not in columns
    engine.dispose()
