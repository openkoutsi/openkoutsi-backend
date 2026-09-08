"""Unit tests for migration 033 — ``training_plans.completed_at``.

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
from backend.app.models.user_orm import Athlete, TrainingPlan

MIGRATION = "backend.app.db.migrations.user.versions.033_plan_completed_at"

_TABLE = "training_plans"
_COLUMN = "completed_at"


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
    UserBase.metadata.create_all(
        engine, tables=[Athlete.__table__, TrainingPlan.__table__]
    )
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
    UserBase.metadata.create_all(
        engine, tables=[Athlete.__table__, TrainingPlan.__table__]
    )
    before = _columns(engine, _TABLE)
    _run(engine)
    assert _columns(engine, _TABLE) == before
    engine.dispose()


def test_an_existing_plan_starts_unstamped(legacy):
    """No backfill on purpose.

    A plan that ended months ago has no record of when anyone noticed, and NULL
    is what invites the auto-close pass to close it — with today's date — the
    first time the athlete's data is read. Stamping every historical plan here
    would instead assert a finish time the database never observed.
    """
    with legacy.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO training_plans "
                "(id, athlete_id, name, start_date, end_date, status, created_at) "
                "VALUES ('plan-1', 'a1', 'Winter block', '2025-01-06', "
                "'2025-03-02', 'active', '2025-01-01')"
            )
        )
    _run(legacy)
    with legacy.connect() as conn:
        row = conn.execute(
            text(f"SELECT status, {_COLUMN} FROM training_plans WHERE id = 'plan-1'")
        ).one()
    assert row[0] == "active"
    assert row[1] is None


def test_downgrade_removes_it_again(legacy):
    _run(legacy)
    _run(legacy, "downgrade")
    assert _COLUMN not in _columns(legacy, _TABLE)
    # And is itself idempotent, so a half-applied downgrade can be retried.
    _run(legacy, "downgrade")


def test_it_chains_from_the_previous_head():
    module = importlib.import_module(MIGRATION)
    assert module.revision == "033_plan_completed_at"
    assert module.down_revision == "032_decoupling_window"
