"""Unit tests for migration 020 — the agentic coach's progress columns (issue #43).

Runs the migration's own ``upgrade()`` against a real SQLite file through a real
Alembic ``Operations`` proxy, so the DDL is exercised rather than paraphrased.

The property that matters most here is idempotence, and it is not academic: new
per-user databases are built by ``create_all`` from the ORM metadata — which
already has these columns — and are never stamped, so the entrypoint replays
001 → head against a DB that already has them. A migration that assumed
otherwise would break every existing user at once.
"""
import importlib
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text

from backend.app.db.base import UserBase
from backend.app.models.user_orm import Activity, Athlete

MIGRATION = "backend.app.db.migrations.user.versions.020_agent_progress"

_ADDED = (("athletes", "training_status_progress"), ("activities", "analysis_progress"))


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
    """A DB built from the ORM with the two new columns removed again.

    Building from the metadata and then dropping is how the fixture stays honest:
    a hand-written CREATE TABLE would drift from the real table the day someone
    adds an unrelated column.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'user.db'}")
    UserBase.metadata.create_all(engine, tables=[Athlete.__table__, Activity.__table__])
    with engine.begin() as conn:
        for table, column in _ADDED:
            conn.execute(text(f'ALTER TABLE "{table}" DROP COLUMN "{column}"'))
    yield engine
    engine.dispose()


def test_upgrade_adds_both_columns(legacy):
    _run(legacy)
    for table, column in _ADDED:
        assert column in _columns(legacy, table)


def test_upgrade_is_idempotent(legacy):
    # The path every fresh per-user DB takes: create_all put the columns there,
    # then the migration loop runs anyway.
    _run(legacy)
    _run(legacy)
    for table, column in _ADDED:
        assert column in _columns(legacy, table)


def test_upgrade_against_a_current_orm_db_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    UserBase.metadata.create_all(engine, tables=[Athlete.__table__, Activity.__table__])
    before = {t: _columns(engine, t) for t, _ in _ADDED}
    _run(engine)
    assert {t: _columns(engine, t) for t, _ in _ADDED} == before
    engine.dispose()


def test_the_columns_are_nullable_so_existing_rows_survive(legacy):
    """No backfill: a row written before the agentic path has no step to report."""
    with legacy.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO athletes "
                "(id, global_user_id, training_status_status, created_at, updated_at) "
                "VALUES ('a1', 'u1', 'done', '2026-08-01', '2026-08-01')"
            )
        )
    _run(legacy)
    with legacy.connect() as conn:
        value = conn.execute(
            text("SELECT training_status_progress FROM athletes WHERE id = 'a1'")
        ).scalar_one()
    assert value is None


def test_downgrade_removes_them_again(legacy):
    _run(legacy)
    _run(legacy, "downgrade")
    for table, column in _ADDED:
        assert column not in _columns(legacy, table)
    # And is itself idempotent, so a half-applied downgrade can be retried.
    _run(legacy, "downgrade")


def test_it_chains_from_the_previous_head():
    module = importlib.import_module(MIGRATION)
    assert module.revision == "020_agent_progress"
    assert module.down_revision == "019_drop_rest_day_activity_links"
