import logging
import shutil
from functools import lru_cache
from pathlib import Path

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.core.config import settings
from backend.app.db.base import UserBase, _set_wal_mode

log = logging.getLogger(__name__)


@lru_cache(maxsize=256)
def _get_user_engine(user_id: str):
    """Build (and cache) the engine for one user's database.

    Deliberately side-effect-free: it creates no directory and no file, and
    ``create_async_engine`` connects lazily, so nothing touches the disk until
    someone opens a session. Creating a user's database is ``init_user_db``'s
    job. This used to ``mkdir`` the parent, which made a cache miss on any read
    path enough to bring a directory into existence — an unauthenticated caller
    passing an unknown id got one directory and three files out of it (issue
    #102, F-03). Opening a session for a user who has none now fails instead,
    which is what the callers already treat as "no such user".
    """
    db_path = Path(settings.user_db_path(user_id))
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
    """Create all per-user tables in a new user DB (idempotent).

    Imports every model module so all tables bound to ``UserBase`` (the athlete
    profile, all training data, the message inbox and the Koutsi conversations)
    are created.

    This is the only place a user's directory comes into existence — creating
    one is an explicit act here rather than a side effect of any code path that
    happens to ask for an engine.
    """
    import backend.app.models.chat_orm  # noqa: F401
    import backend.app.models.message_orm  # noqa: F401
    import backend.app.models.user_orm  # noqa: F401

    Path(settings.user_db_path(user_id)).parent.mkdir(parents=True, exist_ok=True)
    engine = _get_user_engine(user_id)
    async with engine.begin() as conn:
        await conn.execute(text("PRAGMA journal_mode=WAL"))
        await conn.run_sync(UserBase.metadata.create_all)
        await conn.run_sync(_stamp_at_head)


def _stamp_at_head(connection) -> None:
    """Record that a freshly created database is at the latest revision.

    ``create_all`` writes no ``alembic_version`` row, so without this a new
    user's database claims no revision and the next deploy replays every
    migration against it — harmless, since they are idempotent, but real work
    for nothing, and it defeats the "skip users at head" check in
    ``scripts/migrate_user_dbs.py`` (issue #50). ``docker-entrypoint.sh``
    already does this for a fresh registry database.

    Best-effort: if it fails, the next migration run replays rather than skips.
    """
    from alembic.migration import MigrationContext

    try:
        script_dir = _user_script_directory()
        head = script_dir.get_current_head()
        if head is None:
            return
        MigrationContext.configure(connection).stamp(script_dir, head)
    except Exception:
        log.warning(
            "Could not stamp a new database for user at head — the next "
            "migration run will replay instead of skipping",
            exc_info=True,
        )


@lru_cache(maxsize=1)
def _user_script_directory():
    """The per-user migration tree, read once.

    Cached because this is on the path of every new user database rather than
    once per deploy.
    """
    from pathlib import Path as _Path

    from alembic.config import Config
    from alembic.script import ScriptDirectory

    repo_root = _Path(__file__).resolve().parents[3]
    return ScriptDirectory.from_config(
        Config(str(repo_root / "backend" / "alembic-user.ini"))
    )


async def delete_user_db(user_id: str) -> None:
    """Dispose the user's engine and remove their DB directory entirely.

    Used on account deletion so a user's messages are really gone, not orphaned.
    """
    engine = _get_user_engine(user_id)
    await engine.dispose()
    # Evict cached engines so a re-created user_id gets a fresh engine.
    _get_user_engine.cache_clear()
    shutil.rmtree(settings.user_data_dir(user_id), ignore_errors=True)
