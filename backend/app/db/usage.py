"""Engine / session / init for the dedicated LLM-usage database (issue #9).

Kept entirely separate from the registry DB: its own engine, sessionmaker and
Alembic chain. Usage rows are append-only and high-volume, so isolating them
lets the hoster prune/rotate the file independently.

The engine is built lazily and cached by database path (like the per-user
engines) so tests that repoint ``settings.data_dir`` get an isolated usage DB.
"""
import logging
from collections.abc import AsyncGenerator
from functools import lru_cache
from pathlib import Path

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.core.config import settings
from backend.app.db.base import UsageBase, _set_wal_mode


log = logging.getLogger(__name__)

_ALEMBIC_INI = "backend/alembic-usage.ini"
_CHAIN = "llm_usage.db"


@lru_cache(maxsize=8)
def _get_usage_engine(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        echo=False,
        pool_size=3,
        max_overflow=2,
        connect_args={"timeout": 30},
    )
    event.listen(engine.sync_engine, "connect", _set_wal_mode)
    return engine


def usage_session_factory() -> async_sessionmaker:
    return async_sessionmaker(
        _get_usage_engine(settings.llm_usage_db_path), expire_on_commit=False
    )


async def get_usage_session() -> AsyncGenerator[AsyncSession, None]:
    async with usage_session_factory()() as session:
        yield session


async def init_usage_db() -> None:
    """Create the usage table (idempotent — safe on every startup)."""
    import backend.app.models.usage_orm  # noqa: F401

    engine = _get_usage_engine(settings.llm_usage_db_path)
    async with engine.begin() as conn:
        await conn.execute(text("PRAGMA journal_mode=WAL"))
        await conn.run_sync(UsageBase.metadata.create_all)
        await conn.run_sync(_stamp_at_head)


def _stamp_at_head(connection) -> None:
    """Record that a freshly created database is at the latest revision.

    ``create_all`` writes no ``alembic_version`` row, so without this the
    database claims no revision at all and the ``alembic ... upgrade head`` that
    `DEPLOY.md` documents replays ``001`` against tables that already exist —
    failing with "table already exists" on every deployment that has started the
    app once, which is every deployment. Stamping makes the documented command a
    no-op on a database the app created, and a real upgrade on one it did not.

    Mirrors ``db/user_session.py::_stamp_at_head``, and best-effort for the same
    reason: a failure here costs a replay, not data.
    """
    from alembic.migration import MigrationContext

    try:
        script_dir = _script_directory()
        head = script_dir.get_current_head()
        if head is None:
            return
        MigrationContext.configure(connection).stamp(script_dir, head)
    except Exception:
        log.warning(
            "Could not stamp %s at head — `alembic upgrade head` will replay "
            "rather than skip",
            _CHAIN,
            exc_info=True,
        )


@lru_cache(maxsize=1)
def _script_directory():
    """The migration tree for this chain, read once."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    return ScriptDirectory.from_config(Config(_ALEMBIC_INI))
