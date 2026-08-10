"""Unit tests for migration 021 — ``activities.analysis_updated_at`` (issue #91).

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

MIGRATION = "backend.app.db.migrations.user.versions.021_activity_analysis_updated_at"

_TABLE = "activities"
_COLUMN = "analysis_updated_at"


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


def test_existing_rows_survive_with_a_null_clock(legacy):
    """No backfill on purpose.

    A row stranded in ``pending`` before this column existed genuinely has no
    known last-progress time, and ``pending_timed_out`` reads NULL as timed out
    — which is exactly the right answer for a run whose process is long gone.
    """
    with legacy.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO activities (id, athlete_id, status, analysis_status, created_at) "
                "VALUES ('act-1', 'a1', 'processed', 'pending', '2026-08-01')"
            )
        )
    _run(legacy)
    with legacy.connect() as conn:
        value = conn.execute(
            text(f"SELECT {_COLUMN} FROM activities WHERE id = 'act-1'")
        ).scalar_one()
    assert value is None

    from backend.app.services.stranded_runs import pending_timed_out

    assert pending_timed_out(value) is True


def test_downgrade_removes_it_again(legacy):
    _run(legacy)
    _run(legacy, "downgrade")
    assert _COLUMN not in _columns(legacy, _TABLE)
    # And is itself idempotent, so a half-applied downgrade can be retried.
    _run(legacy, "downgrade")


def test_it_chains_from_the_previous_head():
    module = importlib.import_module(MIGRATION)
    assert module.revision == "021_activity_analysis_updated_at"
    assert module.down_revision == "020_agent_progress"
