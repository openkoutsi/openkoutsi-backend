"""Unit tests for migration 031 — the garage (issue #64).

Runs the migration's own ``upgrade()`` against a real SQLite file through a real
Alembic ``Operations`` proxy, so the DDL is exercised rather than paraphrased.

Idempotence is the property that matters most, and it is not academic: new
per-user databases are built by ``create_all`` from the ORM metadata — which
already has every column and table here — and are never stamped, so the
entrypoint replays 001 → head against a DB that already has them.

The other property under test is that this migration **adds nothing to any
existing row**. Every new column lands NULL, which reads as "no bike assigned"
and "no baseline" — both correct for everything that exists when it arrives.
Attaching history is an explicit athlete request, never a migration.
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
    Athlete,
    Bike,
    BikeAccessory,
    BikeMaintenance,
)

MIGRATION = "backend.app.db.migrations.user.versions.031_garage"

NEW_TABLES = {"bike_maintenance", "bike_accessories"}
NEW_BIKE_COLUMNS = {"odometer_base_km", "default_sports", "retired_at"}
NEW_ACTIVITY_COLUMNS = {"bike_id", "bike_source"}

# The tables as they stood *before* this migration. The ORM is always at head,
# so a "legacy" fixture has to *remove* what 031 adds rather than pretend the
# tables did not exist.
_LEGACY_TABLES = [Athlete.__table__, Bike.__table__]

# `activities` is built by hand rather than by dropping columns off the ORM
# table: SQLite refuses to drop a column named in a foreign key, and `bike_id`
# is one. Only the columns this migration reads or writes are needed — it adds
# two columns and an index and touches nothing else — and hand-copying the
# other forty-odd would be a second definition of the schema, rotting quietly
# beside the real one.
_LEGACY_ACTIVITIES = """
CREATE TABLE activities (
    id VARCHAR NOT NULL PRIMARY KEY,
    athlete_id VARCHAR,
    sport_type VARCHAR,
    distance_m FLOAT,
    status VARCHAR
)
"""


def _tables(engine) -> set[str]:
    with engine.connect() as conn:
        return {
            row[0]
            for row in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        }


def _indexes(engine) -> set[str]:
    with engine.connect() as conn:
        return {
            row[0]
            for row in conn.execute(text("SELECT name FROM sqlite_master WHERE type='index'"))
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
    """A DB as it stood before this migration: bikes and activities, no garage."""
    engine = create_engine(f"sqlite:///{tmp_path / 'user.db'}")
    UserBase.metadata.create_all(engine, tables=_LEGACY_TABLES)
    with engine.begin() as conn:
        for column in NEW_BIKE_COLUMNS:
            conn.execute(text(f'ALTER TABLE bikes DROP COLUMN "{column}"'))
        conn.execute(text(_LEGACY_ACTIVITIES))
    yield engine
    engine.dispose()


def test_upgrade_adds_the_columns_and_the_tables(legacy):
    _run(legacy)
    assert NEW_BIKE_COLUMNS <= _columns(legacy, "bikes")
    assert NEW_ACTIVITY_COLUMNS <= _columns(legacy, "activities")
    assert NEW_TABLES <= _tables(legacy)


def test_the_bike_id_index_is_created(legacy):
    """Not optional: every lifetime-distance figure is a SUM filtered on it, so
    without the index each one is a full scan of the athlete's history."""
    _run(legacy)
    assert "ix_activities_bike_id" in _indexes(legacy)


def test_existing_rows_are_left_unassigned(legacy):
    """The migration backfills nothing. NULL means "no bike", which is exactly
    what is true of every ride that predates the feature."""
    with legacy.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO activities (id, athlete_id, sport_type, distance_m, status) "
                "VALUES ('a1', 'ath', 'Ride', 20000.0, 'processed')"
            )
        )
    _run(legacy)
    with legacy.connect() as conn:
        row = conn.execute(
            text("SELECT bike_id, bike_source, distance_m FROM activities WHERE id='a1'")
        ).one()
    assert row[0] is None
    assert row[1] is None
    assert row[2] == 20000.0


def test_upgrade_is_idempotent(legacy):
    _run(legacy)
    _run(legacy)
    assert NEW_TABLES <= _tables(legacy)
    assert NEW_BIKE_COLUMNS <= _columns(legacy, "bikes")


def test_upgrade_against_a_current_orm_db_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    UserBase.metadata.create_all(
        engine,
        tables=[
            *_LEGACY_TABLES,
            BikeMaintenance.__table__,
            BikeAccessory.__table__,
        ],
    )
    before = (_tables(engine), _columns(engine, "bikes"), _columns(engine, "activities"))
    _run(engine)
    after = (_tables(engine), _columns(engine, "bikes"), _columns(engine, "activities"))
    assert before == after
    engine.dispose()


def test_migrated_schema_matches_the_orm(legacy):
    """A row shaped by the ORM fits the migrated table, both directions."""
    _run(legacy)
    for model in (Bike, BikeMaintenance, BikeAccessory):
        orm_columns = {c.name for c in model.__table__.columns}
        assert orm_columns == _columns(legacy, model.__tablename__)
    # `activities` is compared only on what this migration adds — the legacy
    # fixture builds a deliberately minimal one; see `_LEGACY_ACTIVITIES`.
    orm_bike_columns = {
        c.name for c in Activity.__table__.columns if c.name.startswith("bike")
    }
    assert orm_bike_columns == NEW_ACTIVITY_COLUMNS <= _columns(legacy, "activities")


def test_downgrade_removes_them_again(legacy):
    _run(legacy)
    _run(legacy, "downgrade")
    assert not (NEW_TABLES & _tables(legacy))
    assert not (NEW_BIKE_COLUMNS & _columns(legacy, "bikes"))
    assert not (NEW_ACTIVITY_COLUMNS & _columns(legacy, "activities"))
    # And is itself idempotent, so a half-applied downgrade can be retried.
    _run(legacy, "downgrade")


def test_it_chains_from_the_previous_head():
    module = importlib.import_module(MIGRATION)
    assert module.revision == "031_garage"
    assert module.down_revision == "030_course_surface"
