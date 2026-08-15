"""Unit tests for migration 024 — ``import_jobs`` and ``activity_sources.format`` (issue #36).

Runs the migration's own ``upgrade()`` against a real SQLite file through a real
Alembic ``Operations`` proxy, so the DDL is exercised rather than paraphrased.

Idempotence is the property that matters most, and it is not academic: new
per-user databases are built by ``create_all`` from the ORM metadata — which
already has both — and are never stamped, so the entrypoint replays 001 → head
against a DB that already has them.
"""
import importlib
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text

from backend.app.db.base import UserBase
from backend.app.models.user_orm import Activity, ActivitySource, Athlete, ImportJob

MIGRATION = "backend.app.db.migrations.user.versions.024_import_jobs"

_TABLES = [Athlete.__table__, Activity.__table__, ActivitySource.__table__]


def _columns(engine, table: str) -> set[str]:
    with engine.connect() as conn:
        return {row[1] for row in conn.execute(text(f'PRAGMA table_info("{table}")'))}


def _tables(engine) -> set[str]:
    with engine.connect() as conn:
        return {
            row[0]
            for row in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        }


def _run(engine, direction: str = "upgrade") -> None:
    module = importlib.import_module(MIGRATION)
    with engine.begin() as conn:
        operations = Operations(MigrationContext.configure(conn))
        with patch.object(module, "op", operations):
            getattr(module, direction)()


@pytest.fixture
def legacy(tmp_path):
    """A DB as it stood before this migration: no import_jobs, no format column."""
    engine = create_engine(f"sqlite:///{tmp_path / 'user.db'}")
    UserBase.metadata.create_all(engine, tables=_TABLES)
    with engine.begin() as conn:
        conn.execute(text('ALTER TABLE "activity_sources" DROP COLUMN "format"'))
    yield engine
    engine.dispose()


def test_upgrade_creates_the_job_table(legacy):
    _run(legacy)
    assert "import_jobs" in _tables(legacy)


def test_upgrade_adds_the_format_column(legacy):
    _run(legacy)
    assert "format" in _columns(legacy, "activity_sources")


def test_existing_files_are_backfilled_as_fit(legacy):
    """Before this migration there was no other kind of file to have stored."""
    with legacy.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO activities (id, athlete_id, status, created_at) "
                "VALUES ('act-1', 'ath-1', 'processed', '2026-08-01')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO activity_sources (id, activity_id, provider, fit_file_path, "
                "fit_file_encrypted, created_at) "
                "VALUES ('src-1', 'act-1', 'upload', '/data/x.fit', 0, '2026-08-01')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO activity_sources (id, activity_id, provider, fit_file_encrypted, "
                "created_at) VALUES ('src-2', 'act-1', 'strava', 0, '2026-08-01')"
            )
        )

    _run(legacy)

    with legacy.connect() as conn:
        rows = dict(
            conn.execute(text("SELECT id, format FROM activity_sources")).fetchall()
        )
    assert rows["src-1"] == "fit"
    # A source with no file has no format to state.
    assert rows["src-2"] is None


def test_upgrade_is_idempotent(legacy):
    # The path every fresh per-user DB takes: create_all built both, then the
    # migration loop runs anyway.
    _run(legacy)
    _run(legacy)
    assert "import_jobs" in _tables(legacy)
    assert "format" in _columns(legacy, "activity_sources")


def test_upgrade_against_a_current_orm_db_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    UserBase.metadata.create_all(
        engine, tables=[*_TABLES, ImportJob.__table__]
    )
    before = (_tables(engine), _columns(engine, "activity_sources"))
    _run(engine)
    assert (_tables(engine), _columns(engine, "activity_sources")) == before
    engine.dispose()


def test_downgrade_removes_both_again(legacy):
    _run(legacy)
    _run(legacy, "downgrade")
    assert "import_jobs" not in _tables(legacy)
    assert "format" not in _columns(legacy, "activity_sources")
    # And is itself idempotent, so a half-applied downgrade can be retried.
    _run(legacy, "downgrade")


def test_it_chains_from_the_previous_head():
    module = importlib.import_module(MIGRATION)
    assert module.revision == "024_import_jobs"
    assert module.down_revision == "023_sync_leases"
