"""Engine / session / init for the dedicated LLM-usage database (issue #9).

Kept entirely separate from the registry DB: its own engine, sessionmaker and
Alembic chain. Usage rows are append-only and high-volume, so isolating them
lets the hoster prune/rotate the file independently.

The engine is built lazily and cached by database path (like the per-user
engines) so tests that repoint ``settings.data_dir`` get an isolated usage DB.
"""
from collections.abc import AsyncGenerator
from functools import lru_cache
from pathlib import Path

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.core.config import settings
from backend.app.db.base import UsageBase, _set_wal_mode


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
