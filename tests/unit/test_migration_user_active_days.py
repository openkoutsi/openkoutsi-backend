"""Registry migration 020 — the daily-active-users table."""
import importlib
from datetime import date
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text

from backend.app.db.base import RegistryBase
from backend.app.models.registry_orm import User

MIGRATION = "backend.app.db.migrations.registry.versions.020_user_active_days"


def _run(engine, direction: str) -> None:
    module = importlib.import_module(MIGRATION)
    with engine.begin() as conn:
        operations = Operations(MigrationContext.configure(conn))
        with patch.object(module, "op", operations):
            getattr(module, direction)()


def test_revision_chain():
    module = importlib.import_module(MIGRATION)
    assert module.revision == "020_user_active_days"
    assert module.down_revision == "019_signup_halt"


def test_upgrade_matches_the_orm_and_downgrade_removes_it(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'registry.db'}")
    RegistryBase.metadata.create_all(engine, tables=[User.__table__])

    _run(engine, "upgrade")
    insp = inspect(engine)
    assert {c["name"] for c in insp.get_columns("user_active_days")} == {"user_id", "day"}
    assert insp.get_pk_constraint("user_active_days")["constrained_columns"] == [
        "user_id",
        "day",
    ]
    orm = RegistryBase.metadata.tables["user_active_days"]
    assert {c.name for c in orm.columns} == {"user_id", "day"}

    with engine.begin() as conn:
        conn.execute(text("INSERT INTO users (id, password_hash, roles, token_version, created_at) VALUES ('u', 'x', '[]', 0, '2026-01-01 00:00:00')"))
        conn.execute(
            text("INSERT INTO user_active_days (user_id, day) VALUES ('u', :d)"),
            {"d": date(2026, 11, 8)},
        )

    _run(engine, "downgrade")
    assert not inspect(engine).has_table("user_active_days")
    engine.dispose()
