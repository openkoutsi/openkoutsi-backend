"""Engine / session / init for the dedicated third-party API-usage DB (issue #66).

A mirror of :mod:`backend.app.db.usage`, and a sibling of it rather than an
extension: its own engine, sessionmaker and Alembic chain. The reason for a
fourth database is the one ``db/usage.py`` already gives for its own existence —
isolating high-volume append-only rows so the hoster can prune/rotate the file
independently — applied to rows whose volume profile is an order of magnitude
away from the LLM table's, and which therefore want a different retention.

The engine is built lazily and cached by database path (like the per-user
engines) so tests that repoint ``settings.data_dir`` get an isolated DB.
"""
from collections.abc import AsyncGenerator
from functools import lru_cache
from pathlib import Path

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.core.config import settings
from backend.app.db.base import ApiUsageBase, _set_wal_mode


@lru_cache(maxsize=8)
def _get_api_usage_engine(db_path: str):
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


def api_usage_session_factory() -> async_sessionmaker:
    return async_sessionmaker(
        _get_api_usage_engine(settings.api_usage_db_path), expire_on_commit=False
    )


async def get_api_usage_session() -> AsyncGenerator[AsyncSession, None]:
    async with api_usage_session_factory()() as session:
        yield session


async def init_api_usage_db() -> None:
    """Create the api_usage table (idempotent — safe on every startup)."""
    import backend.app.models.api_usage_orm  # noqa: F401

    engine = _get_api_usage_engine(settings.api_usage_db_path)
    async with engine.begin() as conn:
        await conn.execute(text("PRAGMA journal_mode=WAL"))
        await conn.run_sync(ApiUsageBase.metadata.create_all)
