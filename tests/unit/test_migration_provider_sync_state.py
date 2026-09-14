"""Unit tests for migration 035 — ``provider_sync_states`` (issue #68).

Runs the migration's own ``upgrade()`` against a real SQLite file through a real
Alembic ``Operations`` proxy, so the DDL is exercised rather than paraphrased.

Idempotence is the property that matters most, and it is not academic: new
per-user databases are built by ``create_all`` from the ORM metadata — which
already has this table — and are never stamped, so the entrypoint replays
001 → head against a DB that already has it.
"""
import importlib
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text

from backend.app.db.base import UserBase
from backend.app.models.user_orm import ProviderSyncState

MIGRATION = "backend.app.db.migrations.user.versions.035_provider_sync_state"

_TABLE = "provider_sync_states"


def _tables(engine) -> set[str]:
    with engine.connect() as conn:
        return {
            row[0]
            for row in conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
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
    """A DB as it stood before this migration: no sync-state table at all."""
    engine = create_engine(f"sqlite:///{tmp_path / 'user.db'}")
    UserBase.metadata.create_all(engine, tables=[ProviderSyncState.__table__])
    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE "{_TABLE}"'))
    yield engine
    engine.dispose()


def test_upgrade_creates_the_table(legacy):
    _run(legacy)
    assert _TABLE in _tables(legacy)


def test_the_table_carries_every_column_the_model_declares(legacy):
    """A migrated database and a freshly created one must not drift apart."""
    _run(legacy)
    assert _columns(legacy, _TABLE) == {
        column.name for column in ProviderSyncState.__table__.columns
    }


def test_a_row_can_be_written_and_read_back(legacy):
    _run(legacy)
    with legacy.begin() as conn:
        conn.execute(
            text(
                f'INSERT INTO "{_TABLE}" '
                "(provider, status, stop_reason, resume_page, updated_at) "
                "VALUES ('strava', 'stopped', 'throttled', 7, '2026-09-10')"
            )
        )
        row = conn.execute(
            text(f'SELECT status, stop_reason, resume_page, imported FROM "{_TABLE}"')
        ).one()
    # `imported` defaults server-side, so a row written without it is 0 rather
    # than NULL — which is what the API reads straight out.
    assert row == ("stopped", "throttled", 7, 0)


def test_upgrade_is_idempotent(legacy):
    """A fresh create_all DB already has the table and is never stamped."""
    _run(legacy)
    _run(legacy)
    assert _TABLE in _tables(legacy)


def test_downgrade_round_trips(legacy):
    _run(legacy)
    _run(legacy, "downgrade")
    assert _TABLE not in _tables(legacy)

    _run(legacy, "downgrade")  # also idempotent
    assert _TABLE not in _tables(legacy)

    _run(legacy)
    assert _TABLE in _tables(legacy)


def test_it_chains_from_the_previous_head():
    module = importlib.import_module(MIGRATION)
    assert module.revision == "035_provider_sync_state"
    assert module.down_revision == "034_activity_source_streams_fetched_at"
