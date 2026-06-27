from functools import lru_cache
from pathlib import Path

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.core.config import settings
from backend.app.db.base import TeamBase, _set_wal_mode


@lru_cache(maxsize=256)
def _get_team_engine(team_id: str):
    db_path = Path(settings.team_db_path(team_id))
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


def get_team_session_factory(team_id: str) -> async_sessionmaker:
    return async_sessionmaker(_get_team_engine(team_id), expire_on_commit=False)


async def get_team_session(team_id: str) -> AsyncSession:
    """Return a new async session for the given team's DB.

    Callers are responsible for closing it (use as async context manager).
    For use in background tasks — route handlers should use get_ctx_and_session
    from backend.app.core.deps instead.
    """
    return get_team_session_factory(team_id)()


async def init_team_db(team_id: str) -> None:
    """Create all team tables in a new team DB (idempotent)."""
    import backend.app.models.team_orm  # noqa: F401

    engine = _get_team_engine(team_id)
    async with engine.begin() as conn:
        await conn.execute(text("PRAGMA journal_mode=WAL"))
        await conn.run_sync(TeamBase.metadata.create_all)
