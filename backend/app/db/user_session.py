import shutil
from functools import lru_cache
from pathlib import Path

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.core.config import settings
from backend.app.db.base import UserBase, _set_wal_mode


@lru_cache(maxsize=256)
def _get_user_engine(user_id: str):
    db_path = Path(settings.user_db_path(user_id))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        echo=False,
        pool_size=3,
        max_overflow=2,
        # timeout is passed to sqlite3.connect() → reliable busy_timeout in seconds
        connect_args={"timeout": 30},
    )
    event.listen(engine.sync_engine, "connect", _set_wal_mode)  # sets WAL mode
    return engine


def get_user_session_factory(user_id: str) -> async_sessionmaker:
    return async_sessionmaker(_get_user_engine(user_id), expire_on_commit=False)


async def get_user_session(user_id: str) -> AsyncSession:
    """Return a new async session for the given user's DB.

    Callers are responsible for closing it (use as async context manager).
    """
    return get_user_session_factory(user_id)()


async def init_user_db(user_id: str) -> None:
    """Create all per-user tables in a new user DB (idempotent)."""
    import backend.app.models.message_orm  # noqa: F401

    engine = _get_user_engine(user_id)
    async with engine.begin() as conn:
        await conn.execute(text("PRAGMA journal_mode=WAL"))
        await conn.run_sync(UserBase.metadata.create_all)


async def delete_user_db(user_id: str) -> None:
    """Dispose the user's engine and remove their DB directory entirely.

    Used on account deletion so a user's messages are really gone, not orphaned.
    """
    engine = _get_user_engine(user_id)
    await engine.dispose()
    # Evict cached engines so a re-created user_id gets a fresh engine.
    _get_user_engine.cache_clear()
    shutil.rmtree(settings.user_data_dir(user_id), ignore_errors=True)
