"""Unit tests for migration 025 — bikes, courses, course_tracks, course_segments (issue #55).

Runs the migration's own ``upgrade()`` against a real SQLite file through a real
Alembic ``Operations`` proxy, so the DDL is exercised rather than paraphrased.

Idempotence is the property that matters most, and it is not academic: new
per-user databases are built by ``create_all`` from the ORM metadata — which
already has all four tables — and are never stamped, so the entrypoint replays
001 → head against a DB that already has them.
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

MIGRATION = "backend.app.db.migrations.user.versions.025_courses_bikes"

NEW_TABLES = {"bikes", "courses", "course_tracks", "course_segments"}

_LEGACY_TABLES = [Athlete.__table__, Goal.__table__]


def _tables(engine) -> set[str]:
    with engine.connect() as conn:
        return {
            row[0]
            for row in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        }


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
    """A DB as it stood before this migration: athletes and goals, no courses."""
    engine = create_engine(f"sqlite:///{tmp_path / 'user.db'}")
    UserBase.metadata.create_all(engine, tables=_LEGACY_TABLES)
    yield engine
    engine.dispose()


def test_upgrade_creates_all_four_tables(legacy):
    _run(legacy)
    assert NEW_TABLES <= _tables(legacy)


def test_the_course_row_carries_no_coordinate_shaped_column(legacy):
    """The invariant of the storage split: coordinates live in course_tracks
    (and the encrypted blob) only — courses and course_segments must not grow
    a column that could carry one."""
    _run(legacy)
    for table in ("courses", "course_segments"):
        for column in _columns(legacy, table):
            assert "lat" not in column.lower()
            assert "lon" not in column.lower()


def test_upgrade_is_idempotent(legacy):
    _run(legacy)
    _run(legacy)
    assert NEW_TABLES <= _tables(legacy)


def test_upgrade_against_a_current_orm_db_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    UserBase.metadata.create_all(
        engine,
        tables=[
            *_LEGACY_TABLES,
            Bike.__table__,
            Course.__table__,
            CourseTrack.__table__,
            CourseSegment.__table__,
        ],
    )
    before = (_tables(engine), _columns(engine, "courses"))
    _run(engine)
    assert (_tables(engine), _columns(engine, "courses")) == before
    engine.dispose()


def test_migrated_schema_matches_the_orm(legacy):
    """A row shaped by the ORM fits the migrated table, both directions."""
    _run(legacy)
    for model in (Bike, Course, CourseTrack, CourseSegment):
        orm_columns = {c.name for c in model.__table__.columns}
        assert orm_columns == _columns(legacy, model.__tablename__)


def test_downgrade_removes_them_again(legacy):
    _run(legacy)
    _run(legacy, "downgrade")
    assert not (NEW_TABLES & _tables(legacy))
    # And is itself idempotent, so a half-applied downgrade can be retried.
    _run(legacy, "downgrade")


def test_it_chains_from_the_previous_head():
    module = importlib.import_module(MIGRATION)
    assert module.revision == "025_courses_bikes"
    assert module.down_revision == "024_import_jobs"
