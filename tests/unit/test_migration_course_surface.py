"""Unit tests for migration 030 — the surface columns (issue #56).

Same shape as the 026 tests: the migration's own ``upgrade()`` runs against a
real SQLite file through a real Alembic ``Operations`` proxy, so the DDL is
exercised rather than paraphrased.

Idempotence is the property that matters most, and it is not academic: new
per-user databases are built by ``create_all`` from the ORM metadata — which
already has these columns — and are never stamped, so the entrypoint replays
001 → head against a DB that already has them. The other case this has to
survive is a database created *before* course recon existed at all, where none
of the three tables is there.

The behavioural property this one adds: an existing course must come out
**unmatched**, not wrongly labelled. NULL is how "nobody has looked at the
surface under this course" is spelled, and it is the correct answer for every
course that exists when this lands.
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

MIGRATION = "backend.app.db.migrations.user.versions.030_course_surface"

_ADDED = {
    "course_segments": ["surface", "surface_confidence", "surface_raw", "crr_used"],
    "courses": [
        "surface_status",
        "surface_run_id",
        "surface_updated_at",
        "surface_ribbon",
    ],
    "course_tracks": ["surfaces", "surface_matched_at"],
}

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
def without_the_columns(tmp_path):
    """A DB as it stood before this migration: courses, but no surface data."""
    engine = create_engine(f"sqlite:///{tmp_path / 'user.db'}")
    UserBase.metadata.create_all(engine, tables=_COURSE_TABLES)
    with engine.begin() as conn:
        for table, columns in _ADDED.items():
            for column in columns:
                conn.execute(text(f'ALTER TABLE {table} DROP COLUMN "{column}"'))
    yield engine
    engine.dispose()


def test_upgrade_adds_every_column(without_the_columns):
    for table, columns in _ADDED.items():
        assert not set(columns) & _columns(without_the_columns, table)
    _run(without_the_columns)
    for table, columns in _ADDED.items():
        assert set(columns) <= _columns(without_the_columns, table)


def test_upgrade_is_idempotent(without_the_columns):
    _run(without_the_columns)
    _run(without_the_columns)
    for table, columns in _ADDED.items():
        assert set(columns) <= _columns(without_the_columns, table)


def test_upgrade_against_a_current_orm_db_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    UserBase.metadata.create_all(engine, tables=_COURSE_TABLES)
    before = {table: _columns(engine, table) for table in _ADDED}
    _run(engine)
    assert {table: _columns(engine, table) for table in _ADDED} == before
    engine.dispose()


def test_a_database_from_before_course_recon_is_left_alone(tmp_path):
    """001 → head replays on databases that predate the course tables.

    Adding a column to a table that is not there yet must not be an error.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'ancient.db'}")
    UserBase.metadata.create_all(engine, tables=[Athlete.__table__, Goal.__table__])
    _run(engine)
    tables = {
        row[0]
        for row in engine.connect().execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        )
    }
    assert "courses" not in tables
    engine.dispose()


def test_an_existing_course_comes_out_unmatched_rather_than_labelled(
    without_the_columns,
):
    """Nullable and un-backfilled, and that is the honest state.

    Nothing has looked at the road under a course uploaded before this landed,
    so anything other than NULL would be a claim the instance cannot support —
    which is exactly the failure the confidence layer exists to prevent.
    """
    with without_the_columns.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO courses (id, athlete_id, name, gpx_file_key, "
                "gpx_file_encrypted, status, distance_m, created_at, updated_at) "
                "VALUES ('c1', 'a1', 'Old course', 'k', 1, 'ready', 15000.0, "
                "'2026-01-01', '2026-01-01')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO course_segments (id, course_id, segment_index, "
                "start_distance_m, end_distance_m, length_m, avg_gradient, "
                "elevation_change_m, segment_type, speed_capped) VALUES "
                "('s1', 'c1', 0, 0.0, 15000.0, 15000.0, 0.01, 150.0, 'flat', 0)"
            )
        )
    _run(without_the_columns)
    with without_the_columns.connect() as conn:
        course = conn.execute(
            text("SELECT status, surface_status, surface_ribbon FROM courses WHERE id='c1'")
        ).fetchone()
        segment = conn.execute(
            text(
                "SELECT segment_type, surface, surface_confidence, crr_used "
                "FROM course_segments WHERE id='s1'"
            )
        ).fetchone()
    assert course == ("ready", None, None)
    assert segment == ("flat", None, None, None)


def test_downgrade_removes_them_again(without_the_columns):
    _run(without_the_columns)
    _run(without_the_columns, "downgrade")
    for table, columns in _ADDED.items():
        assert not set(columns) & _columns(without_the_columns, table)
    # And is itself idempotent, so a half-applied downgrade can be retried.
    _run(without_the_columns, "downgrade")


def test_it_chains_from_the_previous_head():
    module = importlib.import_module(MIGRATION)
    assert module.revision == "030_course_surface"
    assert module.down_revision == "029_activity_label_suggestions"
