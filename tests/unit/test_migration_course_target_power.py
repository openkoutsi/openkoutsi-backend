"""Unit tests for migration 026 — ``courses.target_power_w`` (issue #61).

Same shape as the 025 tests: the migration's own ``upgrade()`` runs against a
real SQLite file through a real Alembic ``Operations`` proxy, so the DDL is
exercised rather than paraphrased.

Idempotence is the property that matters most, and it is not academic: new
per-user databases are built by ``create_all`` from the ORM metadata — which
already has the column — and are never stamped, so the entrypoint replays
001 → head against a DB that already has it. The other case this has to
survive is a database created *before* course recon existed at all, where
``courses`` is simply absent.
"""
import importlib
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text

from backend.app.db.base import UserBase
from backend.app.models.user_orm import (
    Athlete,
    Bike,
    Course,
    CourseSegment,
    CourseTrack,
    Goal,
)

MIGRATION = "backend.app.db.migrations.user.versions.026_course_target_power"

_COLUMN = "target_power_w"
_COURSE_TABLES = [
    Athlete.__table__,
    Goal.__table__,
    Bike.__table__,
    Course.__table__,
    CourseTrack.__table__,
    CourseSegment.__table__,
]


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
def without_the_column(tmp_path):
    """A DB as it stood before this migration: courses, but no target power."""
    engine = create_engine(f"sqlite:///{tmp_path / 'user.db'}")
    UserBase.metadata.create_all(engine, tables=_COURSE_TABLES)
    with engine.begin() as conn:
        conn.execute(text(f'ALTER TABLE courses DROP COLUMN "{_COLUMN}"'))
    yield engine
    engine.dispose()


def test_upgrade_adds_the_column(without_the_column):
    assert _COLUMN not in _columns(without_the_column, "courses")
    _run(without_the_column)
    assert _COLUMN in _columns(without_the_column, "courses")


def test_upgrade_is_idempotent(without_the_column):
    _run(without_the_column)
    _run(without_the_column)
    assert _COLUMN in _columns(without_the_column, "courses")


def test_upgrade_against_a_current_orm_db_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    UserBase.metadata.create_all(engine, tables=_COURSE_TABLES)
    before = _columns(engine, "courses")
    _run(engine)
    assert _columns(engine, "courses") == before
    engine.dispose()


def test_a_database_from_before_course_recon_is_left_alone(tmp_path):
    """001 → head replays on databases that predate `courses` entirely; adding
    a column to a table that is not there yet must not be an error."""
    engine = create_engine(f"sqlite:///{tmp_path / 'ancient.db'}")
    UserBase.metadata.create_all(engine, tables=[Athlete.__table__, Goal.__table__])
    _run(engine)
    assert "courses" not in {
        row[0]
        for row in engine.connect().execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        )
    }
    engine.dispose()


def test_an_existing_course_keeps_its_time_target(without_the_column):
    """Nullable and un-backfilled: every course that predates this was solved
    for a time or for nothing, and neither of those is a power target."""
    with without_the_column.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO courses (id, athlete_id, name, gpx_file_key, "
                "gpx_file_encrypted, status, target_time_s, distance_m, created_at, "
                "updated_at) VALUES ('c1', 'a1', 'Old course', 'k', 1, 'ready', 3600, "
                "15000.0, '2026-01-01', '2026-01-01')"
            )
        )
    _run(without_the_column)
    with without_the_column.connect() as conn:
        row = conn.execute(
            text(f"SELECT target_time_s, {_COLUMN} FROM courses WHERE id='c1'")
        ).fetchone()
    assert row == (3600, None)


def test_downgrade_removes_it_again(without_the_column):
    _run(without_the_column)
    _run(without_the_column, "downgrade")
    assert _COLUMN not in _columns(without_the_column, "courses")
    # And is itself idempotent, so a half-applied downgrade can be retried.
    _run(without_the_column, "downgrade")


def test_it_chains_from_the_previous_head():
    module = importlib.import_module(MIGRATION)
    assert module.revision == "026_course_target_power"
    assert module.down_revision == "025_courses_bikes"
