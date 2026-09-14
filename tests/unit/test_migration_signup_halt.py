"""Unit tests for registry migration 019 — the temporary signup halt.

Same shape as ``test_migration_course_recon_toggle``, and for the same reason:
``create_all`` covers a fresh install and the revision covers an existing
volume, and neither alone covers both. The starting schema is the ORM metadata
with this revision's own additions removed again, so the fixture cannot drift
from the database the migration actually meets in production.

Unlike 018, nothing here is visible on upgrade: the flag arrives **off**, so an
existing deployment carries on exactly as it was. That is the property worth
pinning — this migration lands on instances that are happily accepting signups,
and a default of on would close them all at once.

The reason column is nullable on purpose and stays that way: "halted, and the
admin did not write a reason" is a real state the API has to render, not a gap
to be filled with an empty string.
"""
import importlib
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text

from backend.app.db.base import RegistryBase
from backend.app.models.registry_orm import InstanceSettings, User

MIGRATION = "backend.app.db.migrations.registry.versions.019_signup_halt"

_PRE_019_COLUMNS = (
    "id, llm_requires_subscription, allow_self_signup, "
    "allow_personal_access_tokens, allow_mcp_server, allow_course_recon, updated_at"
)


def _seed_pre_019(engine) -> None:
    """The registry as revision 018 left it, with a settings row already in it."""
    RegistryBase.metadata.create_all(
        engine, tables=[User.__table__, InstanceSettings.__table__]
    )
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE instance_settings DROP COLUMN signups_halted"))
        conn.execute(
            text("ALTER TABLE instance_settings DROP COLUMN signup_halt_reason")
        )
        conn.execute(
            text(
                f"INSERT INTO instance_settings ({_PRE_019_COLUMNS}) "
                "VALUES (1, 0, 1, 1, 1, 0, '2026-01-01 00:00:00')"
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
    _seed_pre_019(engine)
    _run(engine)
    yield engine
    engine.dispose()


def test_it_follows_the_course_recon_revision():
    module = importlib.import_module(MIGRATION)
    assert module.revision == "019_signup_halt"
    assert module.down_revision == "018_course_recon_toggle"


def test_both_columns_are_added(migrated):
    columns = {c["name"] for c in inspect(migrated).get_columns("instance_settings")}
    assert {"signups_halted", "signup_halt_reason"} <= columns


def test_an_existing_instance_keeps_accepting_signups(migrated):
    """The property that makes this migration safe to ship.

    It lands on instances currently taking signups. Arriving halted would close
    every one of them, and the admin would find out from their users.
    """
    with migrated.begin() as conn:
        row = conn.execute(
            text(
                "SELECT signups_halted, signup_halt_reason, allow_self_signup "
                "FROM instance_settings WHERE id = 1"
            )
        ).one()
    assert row.signups_halted == 0
    assert row.signup_halt_reason is None
    # And the standing policy it sits beside is untouched.
    assert row.allow_self_signup == 1


def test_the_flag_is_not_nullable(migrated):
    """A tri-state switch has a state nobody chose."""
    column = next(
        c
        for c in inspect(migrated).get_columns("instance_settings")
        if c["name"] == "signups_halted"
    )
    assert column["nullable"] is False


def test_the_reason_is_nullable(migrated):
    """"Halted, no reason given" is a state the sign-up page has to render."""
    column = next(
        c
        for c in inspect(migrated).get_columns("instance_settings")
        if c["name"] == "signup_halt_reason"
    )
    assert column["nullable"] is True


def test_a_row_inserted_without_the_columns_still_defaults_open(migrated):
    with migrated.begin() as conn:
        conn.execute(
            text(
                f"INSERT INTO instance_settings ({_PRE_019_COLUMNS}) "
                "VALUES (2, 0, 0, 1, 1, 0, '2026-02-01 00:00:00')"
            )
        )
        row = conn.execute(
            text(
                "SELECT signups_halted, signup_halt_reason "
                "FROM instance_settings WHERE id = 2"
            )
        ).one()
    assert row.signups_halted == 0
    assert row.signup_halt_reason is None


def test_the_model_and_the_revision_agree(migrated):
    """``create_all`` builds a fresh install and the revision upgrades an old
    one; they have to land in the same place."""
    migrated_columns = {
        c["name"] for c in inspect(migrated).get_columns("instance_settings")
    }
    assert migrated_columns == {c.name for c in InstanceSettings.__table__.columns}


def test_the_downgrade_removes_them(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'registry.db'}")
    _seed_pre_019(engine)
    _run(engine)
    _run(engine, "downgrade")
    columns = {c["name"] for c in inspect(engine).get_columns("instance_settings")}
    assert "signups_halted" not in columns
    assert "signup_halt_reason" not in columns
    engine.dispose()
