"""Unit tests for migration 034 — ``activity_sources.streams_fetched_at`` (issue #67).

Runs the migration's own ``upgrade()`` against a real SQLite file through a real
Alembic ``Operations`` proxy, so the DDL and the backfill are exercised rather
than paraphrased.

The backfill is what most of this covers. Adding the column alone would leave
every existing row NULL, and NULL is the sync's signal to go back to the
provider — so the first sync after this deploys would re-fetch each athlete's
entire history against the shared quota, which is the burst the issue is about.
"""
import importlib
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text

from backend.app.db.base import UserBase
from backend.app.models.user_orm import (
    Activity,
    ActivitySource,
    ActivityStream,
    Athlete,
)

MIGRATION = "backend.app.db.migrations.user.versions.034_activity_source_streams_fetched_at"

_TABLE = "activity_sources"
_COLUMN = "streams_fetched_at"


def _columns(engine, table: str) -> set[str]:
    with engine.connect() as conn:
        return {row[1] for row in conn.execute(text(f'PRAGMA table_info("{table}")'))}


def _run(engine, direction: str = "upgrade") -> None:
    module = importlib.import_module(MIGRATION)
    with engine.begin() as conn:
        operations = Operations(MigrationContext.configure(conn))
        with patch.object(module, "op", operations):
            getattr(module, direction)()


def _fetched_at(engine, source_id: str):
    with engine.connect() as conn:
        return conn.execute(
            text(f"SELECT {_COLUMN} FROM {_TABLE} WHERE id = :i"), {"i": source_id}
        ).scalar_one()


def _seed_activity(conn, activity_id: str) -> None:
    conn.execute(
        text(
            "INSERT INTO activities (id, athlete_id, status, created_at) "
            "VALUES (:i, 'a1', 'processed', '2026-08-01 00:00:00')"
        ),
        {"i": activity_id},
    )


def _seed_source(conn, source_id: str, activity_id: str, fit_path: str | None) -> None:
    conn.execute(
        text(
            "INSERT INTO activity_sources "
            "(id, activity_id, provider, external_id, fit_file_path, "
            " fit_file_encrypted, created_at) "
            "VALUES (:i, :a, 'strava', :e, :f, 0, '2026-08-02 09:30:00')"
        ),
        {"i": source_id, "a": activity_id, "e": f"ext-{source_id}", "f": fit_path},
    )


def _seed_stream(conn, activity_id: str) -> None:
    conn.execute(
        text(
            "INSERT INTO activity_streams (id, activity_id, stream_type, data) "
            "VALUES (:i, :a, 'power', '[1, 2, 3]')"
        ),
        {"i": f"stream-{activity_id}", "a": activity_id},
    )


@pytest.fixture
def legacy(tmp_path):
    """A DB built from the ORM with the new column dropped again.

    Building from the metadata and then dropping keeps the fixture from drifting
    away from the real table the day someone adds an unrelated column.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'user.db'}")
    UserBase.metadata.create_all(
        engine,
        tables=[
            Athlete.__table__,
            Activity.__table__,
            ActivitySource.__table__,
            ActivityStream.__table__,
        ],
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
        engine,
        tables=[
            Athlete.__table__,
            Activity.__table__,
            ActivitySource.__table__,
            ActivityStream.__table__,
        ],
    )
    before = _columns(engine, _TABLE)
    _run(engine)
    assert _columns(engine, _TABLE) == before
    engine.dispose()


def test_a_source_with_a_stored_file_is_stamped(legacy):
    with legacy.begin() as conn:
        _seed_activity(conn, "act-1")
        _seed_source(conn, "src-1", "act-1", "/data/act-1.fit")
    _run(legacy)
    # ``created_at``, not the migration's own clock: it records when the data was
    # fetched, and today is not when that happened.
    assert str(_fetched_at(legacy, "src-1")).startswith("2026-08-02 09:30:00")


def test_a_source_whose_activity_has_streams_is_stamped(legacy):
    with legacy.begin() as conn:
        _seed_activity(conn, "act-2")
        _seed_source(conn, "src-2", "act-2", None)
        _seed_stream(conn, "act-2")
    _run(legacy)
    assert _fetched_at(legacy, "src-2") is not None


def test_a_source_with_neither_is_left_for_repair(legacy):
    """No file and no streams is the shape of an import that fell short.

    Left NULL on purpose: the next sync tries once more and settles it either
    way, which is how athletes already affected are healed rather than only new
    imports being protected.
    """
    with legacy.begin() as conn:
        _seed_activity(conn, "act-3")
        _seed_source(conn, "src-3", "act-3", None)
    _run(legacy)
    assert _fetched_at(legacy, "src-3") is None


def test_one_hollow_source_does_not_stamp_another_activitys(legacy):
    """The EXISTS is per activity, not per database."""
    with legacy.begin() as conn:
        _seed_activity(conn, "act-good")
        _seed_source(conn, "src-good", "act-good", None)
        _seed_stream(conn, "act-good")
        _seed_activity(conn, "act-hollow")
        _seed_source(conn, "src-hollow", "act-hollow", None)
    _run(legacy)
    assert _fetched_at(legacy, "src-good") is not None
    assert _fetched_at(legacy, "src-hollow") is None


def test_downgrade_removes_it_again(legacy):
    _run(legacy)
    _run(legacy, "downgrade")
    assert _COLUMN not in _columns(legacy, _TABLE)
    # And is itself idempotent, so a half-applied downgrade can be retried.
    _run(legacy, "downgrade")


def test_it_chains_from_the_previous_head():
    module = importlib.import_module(MIGRATION)
    assert module.revision == "034_activity_source_streams_fetched_at"
    assert module.down_revision == "033_plan_completed_at"
