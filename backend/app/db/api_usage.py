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
import logging
from collections.abc import AsyncGenerator
from functools import lru_cache
from pathlib import Path

from sqlalchemy import event, inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.core.config import settings
from backend.app.db.base import ApiUsageBase, _set_wal_mode


log = logging.getLogger(__name__)

#: Anchored to this file rather than the working directory — the stamp below
#: runs at startup, wherever the process happens to have been launched from.
_ALEMBIC_INI = str(Path(__file__).resolve().parents[2] / "alembic-api-usage.ini")
_CHAIN = "api_usage.db"


@lru_cache(maxsize=8)
def _get_api_usage_engine(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        echo=False,
        pool_size=3,
        max_overflow=2,
        # Deliberately far shorter than the LLM-usage engine's 30s. Rows here
        # are written once per outbound HTTP request rather than once per
        # deliberate user action, so a database held under an exclusive lock —
        # the documented retention `VACUUM` — would otherwise queue a 30-second
        # task per request for the length of the prune. Shedding the row is the
        # right trade: it is an accounting row, and the sync is not.
        pool_timeout=5,
        connect_args={"timeout": 5},
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
        # Whether this database already existed decides whether stamping it is
        # safe. This runs on *every* startup, not only at creation — unlike
        # ``db/user_session.py``, whose stamp is reached only from
        # ``init_user_db``. Stamping unconditionally would move an existing
        # deployment's revision forward without running the migration, and
        # ``create_all`` adds missing tables but never missing *columns*, so the
        # schema would silently stay behind a version it claims to be at.
        existed = await conn.run_sync(
            lambda sync_conn: inspect(sync_conn).has_table("api_usage")
        )
        await conn.run_sync(ApiUsageBase.metadata.create_all)
        if not existed:
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
