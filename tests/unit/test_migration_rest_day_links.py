"""Unit tests for migration 019 — dropping activity links to rest days (issue #40).

Runs the migration's own ``upgrade()`` against a real SQLite file with
``op.get_bind`` patched to a synchronous connection, so the SQL itself is
exercised rather than a paraphrase of it. The schema comes from the ORM metadata
rather than a hand-written CREATE TABLE, so the fixture cannot drift away from
the table the migration actually meets in production.
"""
import importlib
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text

from backend.app.db.base import UserBase
from backend.app.models.user_orm import (
    Activity,
    PlannedWorkout,
    PlannedWorkoutActivity,
    TrainingPlan,
)

MIGRATION = (
    "backend.app.db.migrations.user.versions.019_drop_rest_day_activity_links"
)

_NOW = datetime(2026, 8, 2, 7, 51, tzinfo=timezone.utc)

# (workout id, workout_type, duration_min, target_load, activity id, survives?)
_ROWS = (
    # Genuine rest days — plan_builder._rest_day leaves both figures NULL.
    ("w-rest", "rest", None, None, "a-sunday-ride", False),
    ("w-rest-caps", "Rest", None, None, "a-caps", False),
    ("w-rest-space", "  rest  ", None, None, "a-spaced", False),
    ("w-rest-tab", "\trest", None, None, "a-tabbed", False),
    ("w-untyped", None, None, None, "a-untyped", False),
    ("w-blank", "", None, None, "a-blank", False),
    # A rest day the LLM gave figures to is still a declared rest day.
    ("w-rest-figures", "rest", 60, 50, "a-rest-with-figures", False),
    # An untyped row carrying a real prescription is a session that lost its
    # type, not a rest day — its link may even have been made by hand.
    ("w-untyped-real", "", 120, 150, "a-prescribed", True),
    ("w-null-real", None, 90, 100, "a-prescribed-null", True),
    # Ordinary sessions.
    ("w-long", "long", 180, 200, "a-saturday-ride", True),
    ("w-recovery", "recovery", 45, 25, "a-recovery-spin", True),
)


def _seed(engine) -> None:
    UserBase.metadata.create_all(
        engine,
        tables=[
            t.__table__
            for t in (Activity, TrainingPlan, PlannedWorkout, PlannedWorkoutActivity)
        ],
    )
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO training_plans (id, athlete_id, name, status, created_at) "
                "VALUES ('p', 'ath', 'P', 'active', :now)"
            ),
            {"now": _NOW},
        )
        for wid, wtype, dur, load, aid, _ in _ROWS:
            conn.execute(
                text(
                    "INSERT INTO planned_workouts "
                    "(id, plan_id, week_number, day_of_week, workout_type, "
                    " duration_min, target_load) "
                    "VALUES (:id, 'p', 1, 1, :t, :d, :l)"
                ),
                {"id": wid, "t": wtype, "d": dur, "l": load},
            )
            conn.execute(
                text(
                    "INSERT INTO activities (id, athlete_id, status, created_at) "
                    "VALUES (:a, 'ath', 'processed', :now)"
                ),
                {"a": aid, "now": _NOW},
            )
            conn.execute(
                text(
                    "INSERT INTO planned_workout_activities "
                    "(planned_workout_id, activity_id) VALUES (:w, :a)"
                ),
                {"w": wid, "a": aid},
            )


def _run_upgrade(engine) -> None:
    module = importlib.import_module(MIGRATION)
    with engine.begin() as conn:
        with patch.object(module.op, "get_bind", return_value=conn):
            module.upgrade()
            # Re-running must be safe: the DELETE is a plain predicate.
            module.upgrade()


def _links(engine) -> set[str]:
    with engine.connect() as conn:
        return {
            r[0]
            for r in conn.execute(
                text("SELECT activity_id FROM planned_workout_activities")
            )
        }


@pytest.fixture
def migrated(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'user.db'}")
    _seed(engine)
    _run_upgrade(engine)
    yield engine
    engine.dispose()


def test_rest_day_links_are_dropped(migrated):
    survivors = _links(migrated)
    assert {aid for *_, aid, keep in _ROWS if not keep}.isdisjoint(survivors)


def test_prescribed_and_ordinary_links_are_kept(migrated):
    assert {aid for *_, aid, keep in _ROWS if keep} == _links(migrated)


def test_nothing_else_is_touched(migrated):
    """The blast radius is one table — no cascade, no collateral rows."""
    with migrated.connect() as conn:
        counts = {
            t: conn.execute(text(f"SELECT count(*) FROM {t}")).scalar_one()
            for t in ("activities", "planned_workouts", "training_plans")
        }
    assert counts == {
        "activities": len(_ROWS),
        "planned_workouts": len(_ROWS),
        "training_plans": 1,
    }


def test_dropped_rows_are_recoverable(migrated):
    """Every deleted row is kept, so the irreversible delete stays undoable."""
    module = importlib.import_module(MIGRATION)
    with migrated.connect() as conn:
        saved = {
            r[0]
            for r in conn.execute(
                text(f"SELECT activity_id FROM {module.BACKUP_TABLE}")
            )
        }
    # Re-running the migration must not lose or duplicate the snapshot either.
    assert saved == {aid for *_, aid, keep in _ROWS if not keep}


def test_upgrade_is_a_noop_without_the_tables(tmp_path):
    """A DB predating the join table must not blow up the migration."""
    engine = create_engine(f"sqlite:///{tmp_path / 'bare.db'}")
    _run_upgrade(engine)
    engine.dispose()


async def test_full_chain_applies_to_a_fresh_user_db(isolate_user_dbs, monkeypatch):
    """`create_all` DB → `alembic upgrade head`: the path every deploy takes.

    New per-user DBs are built by ``init_user_db`` and never stamped, so the
    entrypoint's migration loop replays 001 → head against them. A migration
    that assumes a column the ORM already created (or is missing one it did not)
    breaks every existing user at once, which no per-migration test would catch.
    """
    import asyncio

    from alembic import command
    from alembic.config import Config

    from backend.app.db.user_session import init_user_db

    user_id = "migration-chain-user"
    await init_user_db(user_id)

    # env.py resolves the DB path from USER_ID via settings, which
    # ``isolate_user_dbs`` has already pointed at this test's temp dir.
    monkeypatch.setenv("USER_ID", user_id)
    cfg = Config("backend/alembic-user.ini")
    await asyncio.to_thread(command.upgrade, cfg, "head")
