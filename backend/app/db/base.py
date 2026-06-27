from sqlalchemy.orm import DeclarativeBase


class RegistryBase(DeclarativeBase):
    """Base for the global registry DB (users, invitations, provider connections,
    instance settings)."""
    pass


class UserBase(DeclarativeBase):
    """Base for per-user DBs (one isolated SQLite file per user).

    Holds the athlete profile, all of the user's training data and their message
    inbox. One athlete per DB.
    """
    pass


def _set_wal_mode(dbapi_conn, _connection_record) -> None:
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()
