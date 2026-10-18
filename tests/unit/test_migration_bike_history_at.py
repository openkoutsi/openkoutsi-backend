"""Unit tests for migration 037 — the "Attach past rides" marker.

Runs the migration's own ``upgrade()``/``downgrade()`` against a real SQLite file
through a real Alembic ``Operations`` proxy. Idempotence matters because fresh
per-user databases are built by ``create_all`` and then replayed to head.
"""
import importlib
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text

MIGRATION = "backend.app.db.migrations.user.versions.037_activity_bike_history_at"


def _columns(conn):
    return {row[1] for row in conn.execute(text('PRAGMA table_info("activities")'))}


def _run(conn, fn_name):
    mod = importlib.import_module(MIGRATION)
    ops = Operations(MigrationContext.configure(conn))
    with patch.object(mod, "op", ops):
        getattr(mod, fn_name)()


def test_upgrade_adds_the_column_and_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'u.db'}")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE activities (id VARCHAR PRIMARY KEY, bike_source VARCHAR)"
        ))
        conn.execute(text("INSERT INTO activities (id, bike_source) VALUES ('a', 'auto')"))
        _run(conn, "upgrade")
        _run(conn, "upgrade")
        assert "bike_history_at" in _columns(conn)
        # No backfill: an existing automatic match is not claimed by the pass.
        assert conn.execute(text("SELECT bike_history_at FROM activities")).scalar() is None

        _run(conn, "downgrade")
        _run(conn, "downgrade")
        assert "bike_history_at" not in _columns(conn)
