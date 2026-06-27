from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.core.config import settings
from backend.app.db.base import RegistryBase, _set_wal_mode


def _make_registry_engine():
    Path(settings.registry_db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{settings.registry_db_path}",
        echo=False,
        pool_size=3,
        max_overflow=2,
        connect_args={"timeout": 30},
    )
    event.listen(engine.sync_engine, "connect", _set_wal_mode)
    return engine


_registry_engine = _make_registry_engine()
_RegistrySessionLocal = async_sessionmaker(_registry_engine, expire_on_commit=False)


async def get_registry_session() -> AsyncGenerator[AsyncSession, None]:
    async with _RegistrySessionLocal() as session:
        yield session


async def init_registry_db() -> None:
    """Create all registry tables (idempotent — safe to call on every startup)."""
    import backend.app.models.registry_orm  # noqa: F401

    async with _registry_engine.begin() as conn:
        await conn.execute(text("PRAGMA journal_mode=WAL"))
        await conn.run_sync(RegistryBase.metadata.create_all)
