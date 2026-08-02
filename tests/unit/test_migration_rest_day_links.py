"""Unit test for migration 019 — dropping activity links to rest days (issue #40).

Runs the migration's ``upgrade()`` against a real SQLite file with ``op.get_bind``
patched to a plain synchronous connection, so the SQL itself is exercised rather
than a paraphrase of it.
"""
import importlib
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text

MIGRATION = (
    "backend.app.db.migrations.user.versions.019_drop_rest_day_activity_links"
)

_SCHEMA = (
    """
    CREATE TABLE planned_workouts (
        id TEXT PRIMARY KEY,
        plan_id TEXT,
        week_number INTEGER,
        day_of_week INTEGER,
        workout_type TEXT
    )
    """,
    """
    CREATE TABLE planned_workout_activities (
        planned_workout_id TEXT,
        activity_id TEXT,
        PRIMARY KEY (planned_workout_id, activity_id)
    )
    """,
)

# (workout id, workout_type, activity id, survives the migration?)
_ROWS = (
    ("w-rest", "rest", "a-sunday-ride", False),
    ("w-rest-caps", "Rest", "a-caps", False),
    ("w-untyped", None, "a-untyped", False),
    ("w-blank", "", "a-blank", False),
    ("w-long", "long", "a-saturday-ride", True),
    ("w-recovery", "recovery", "a-recovery-spin", True),
)


@pytest.fixture
def migrated(tmp_path):
    """Seed a DB with the rows above, run upgrade(), return the surviving links."""
    engine = create_engine(f"sqlite:///{tmp_path / 'user.db'}")
    with engine.begin() as conn:
        for stmt in _SCHEMA:
            conn.execute(text(stmt))
        for wid, wtype, aid, _ in _ROWS:
            conn.execute(
                text(
                    "INSERT INTO planned_workouts "
                    "(id, plan_id, week_number, day_of_week, workout_type) "
                    "VALUES (:id, 'p', 1, 1, :t)"
                ),
                {"id": wid, "t": wtype},
            )
            conn.execute(
                text(
                    "INSERT INTO planned_workout_activities "
                    "(planned_workout_id, activity_id) VALUES (:w, :a)"
                ),
                {"w": wid, "a": aid},
            )

    module = importlib.import_module(MIGRATION)
    with engine.begin() as conn:
        with patch.object(module.op, "get_bind", return_value=conn):
            module.upgrade()
            # Re-running must be safe: the DELETE is a plain predicate.
            module.upgrade()

    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT activity_id FROM planned_workout_activities")
        ).fetchall()
    engine.dispose()
    return {r[0] for r in rows}


def test_rest_day_links_are_dropped(migrated):
    assert {aid for _, _, aid, keep in _ROWS if not keep}.isdisjoint(migrated)


def test_real_workout_links_are_kept(migrated):
    assert {aid for _, _, aid, keep in _ROWS if keep} == migrated


def test_upgrade_is_a_noop_without_the_tables(tmp_path):
    """A DB predating the join table must not blow up the migration."""
    engine = create_engine(f"sqlite:///{tmp_path / 'bare.db'}")
    module = importlib.import_module(MIGRATION)
    with engine.begin() as conn:
        with patch.object(module.op, "get_bind", return_value=conn):
            module.upgrade()
    engine.dispose()
