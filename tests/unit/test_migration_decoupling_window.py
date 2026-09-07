"""Unit tests for migration 032 — ``activities.decoupling_window_s``.

Runs the migration's own ``upgrade()`` against a real SQLite file through a real
Alembic ``Operations`` proxy, so the DDL is exercised rather than paraphrased.

Idempotence is the property that matters most, and it is not academic: new
per-user databases are built by ``create_all`` from the ORM metadata — which
already has this column — and are never stamped, so the entrypoint replays
001 → head against a DB that already has it.
"""
import importlib
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text

from backend.app.db.base import UserBase
from backend.app.models.user_orm import Activity, Athlete

MIGRATION = "backend.app.db.migrations.user.versions.032_decoupling_window"

_TABLE = "activities"
_COLUMN = "decoupling_window_s"


def _columns(engine, table: str) -> set[str]:
    with engine.connect() as conn:
        return {row[1] for row in conn.execute(text(f'PRAGMA table_info("{table}")'))}


def _run(engine, direction: str = "upgrade") -> None:
    module = importlib.import_module(MIGRATION)
    with engine.begin() as conn:
        operations = Operations(MigrationContext.configure(conn))
        with patch.object(module, "op", operations):
            getattr(module, direction)()


@pytest.fixture
def legacy(tmp_path):
    """A DB built from the ORM with the new column removed again.

    Building from the metadata and then dropping is how the fixture stays
    honest: a hand-written CREATE TABLE would drift from the real table the day
    someone adds an unrelated column.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'user.db'}")
    UserBase.metadata.create_all(engine, tables=[Athlete.__table__, Activity.__table__])
    with engine.begin() as conn:
        conn.execute(text(f'ALTER TABLE "{_TABLE}" DROP COLUMN "{_COLUMN}"'))
    yield engine
    engine.dispose()


def test_upgrade_adds_the_column(legacy):
    _run(legacy)
    assert _COLUMN in _columns(legacy, _TABLE)


def test_upgrade_is_idempotent(legacy):
    # The path every fresh per-user DB takes: create_all put the column there,
    # then the migration loop runs anyway.
    _run(legacy)
    _run(legacy)
    assert _COLUMN in _columns(legacy, _TABLE)


def test_upgrade_against_a_current_orm_db_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    UserBase.metadata.create_all(engine, tables=[Athlete.__table__, Activity.__table__])
    before = _columns(engine, _TABLE)
    _run(engine)
    assert _columns(engine, _TABLE) == before
    engine.dispose()


def test_an_existing_figure_keeps_its_null_window(legacy):
    """No backfill on purpose.

    A ride processed before this column existed has a figure measured across
    the whole of it, and no record of what "the whole of it" was. NULL says
    "this one predates the window" — which is what the API and the card need to
    know before they say anything about which block was measured. The ride
    picks up a real window on its next reprocess.
    """
    with legacy.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO activities (id, athlete_id, status, decoupling_pct, created_at) "
                "VALUES ('act-1', 'a1', 'processed', 3.4, '2026-08-01')"
            )
        )
    _run(legacy)
    with legacy.connect() as conn:
        row = conn.execute(
            text(f"SELECT decoupling_pct, {_COLUMN} FROM activities WHERE id = 'act-1'")
        ).one()
    assert row[0] == 3.4
    assert row[1] is None


def test_downgrade_removes_it_again(legacy):
    _run(legacy)
    _run(legacy, "downgrade")
    assert _COLUMN not in _columns(legacy, _TABLE)
    # And is itself idempotent, so a half-applied downgrade can be retried.
    _run(legacy, "downgrade")


def test_it_chains_from_the_previous_head():
    module = importlib.import_module(MIGRATION)
    assert module.revision == "032_decoupling_window"
    assert module.down_revision == "031_garage"
