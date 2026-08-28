"""Unit tests for migration 028 — ``athletes.achievements_dirty_at`` (issue #69).

Runs the migration's own ``upgrade()`` against a real SQLite file through a real
Alembic ``Operations`` proxy, so the DDL is exercised rather than paraphrased.

Idempotence is the property that matters most, and it is not academic: new
per-user databases are built by ``create_all`` from the ORM metadata — which
already has the column — and are never stamped, so the entrypoint replays
001 → head against a DB that already has it.
"""
import importlib
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text

from backend.app.db.base import UserBase
from backend.app.models.user_orm import Athlete

MIGRATION = "backend.app.db.migrations.user.versions.028_achievements_dirty"

_COLUMN = "achievements_dirty_at"


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
    """A DB as it stood before this migration: no achievements_dirty_at column."""
    engine = create_engine(f"sqlite:///{tmp_path / 'user.db'}")
    UserBase.metadata.create_all(engine, tables=[Athlete.__table__])
    with engine.begin() as conn:
        conn.execute(text(f'ALTER TABLE "athletes" DROP COLUMN "{_COLUMN}"'))
    yield engine
    engine.dispose()


def test_upgrade_adds_the_column(legacy):
    _run(legacy)
    assert _COLUMN in _columns(legacy, "athletes")


def test_existing_athletes_are_left_clean(legacy):
    """Un-backfilled on purpose: the code this replaces recomputed eagerly, so
    nothing is owed for an athlete who existed before the column did."""
    with legacy.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO athletes (id, global_user_id, created_at, updated_at) "
                "VALUES ('ath-1', 'user-1', '2026-08-01', '2026-08-01')"
            )
        )

    _run(legacy)

    with legacy.connect() as conn:
        value = conn.execute(
            text(f'SELECT "{_COLUMN}" FROM athletes WHERE id = \'ath-1\'')
        ).scalar_one()
    assert value is None


def test_upgrade_is_idempotent(legacy):
    """A fresh create_all DB already has the column and is never stamped."""
    _run(legacy)
    _run(legacy)
    assert _COLUMN in _columns(legacy, "athletes")


def test_downgrade_round_trips(legacy):
    _run(legacy)
    _run(legacy, "downgrade")
    assert _COLUMN not in _columns(legacy, "athletes")

    _run(legacy, "downgrade")  # also idempotent
    assert _COLUMN not in _columns(legacy, "athletes")

    _run(legacy)
    assert _COLUMN in _columns(legacy, "athletes")
