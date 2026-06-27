from sqlalchemy.orm import DeclarativeBase


class RegistryBase(DeclarativeBase):
    """Base for the global registry DB (users, teams, memberships, provider connections)."""
    pass


class TeamBase(DeclarativeBase):
    """Base for per-team DBs (athletes, activities, training data)."""
    pass


class UserBase(DeclarativeBase):
    """Base for per-user DBs (one isolated SQLite file per user).

    Generic on purpose: today it holds the user's message inbox, but other
    per-user data can be added as additional tables in the same file.
    """
    pass


# Keep the legacy alias so any remaining references to `Base` import still resolve
# during the transition. Remove once all callers are updated.
Base = RegistryBase


def _set_wal_mode(dbapi_conn, _connection_record) -> None:
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()
