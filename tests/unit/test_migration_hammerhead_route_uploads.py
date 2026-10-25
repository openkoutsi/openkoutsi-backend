"""Unit tests for migration 038 — the Hammerhead route-upload join table (issue #69).

Runs the migration's own ``upgrade()``/``downgrade()`` against a real SQLite file
through a real Alembic ``Operations`` proxy. Idempotence matters because fresh
per-user databases are built by ``create_all`` and then replayed to head.
"""
import importlib
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text

MIGRATION = "backend.app.db.migrations.user.versions.038_hammerhead_route_uploads"


def _tables(conn):
    return {
        row[0]
        for row in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
    }


def _run(conn, fn_name):
    mod = importlib.import_module(MIGRATION)
    ops = Operations(MigrationContext.configure(conn))
    with patch.object(mod, "op", ops):
        getattr(mod, fn_name)()


def test_upgrade_creates_the_table_and_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'u.db'}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE athletes (id VARCHAR PRIMARY KEY)"))
        conn.execute(text("CREATE TABLE courses (id VARCHAR PRIMARY KEY)"))
        _run(conn, "upgrade")
        _run(conn, "upgrade")
        assert "hammerhead_route_uploads" in _tables(conn)
        columns = {
            row[1]
            for row in conn.execute(text('PRAGMA table_info("hammerhead_route_uploads")'))
        }
        assert {"athlete_id", "course_id", "hammerhead_route_id"} <= columns

        _run(conn, "downgrade")
        _run(conn, "downgrade")
        assert "hammerhead_route_uploads" not in _tables(conn)
