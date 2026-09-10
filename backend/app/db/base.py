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


class UsageBase(DeclarativeBase):
    """Base for the dedicated LLM-usage DB (issue #9).

    A single, separate SQLite file holding one append-only row per
    instance-paid LLM call (BYOK calls are never recorded). Kept apart from the
    registry DB so its high-volume, unbounded rows can be pruned/rotated
    independently and carry no registry foreign keys.
    """
    pass


class ApiUsageBase(DeclarativeBase):
    """Base for the dedicated third-party API-usage DB (issue #66).

    A sibling of :class:`UsageBase`, not an extension of it: one append-only row
    per outbound HTTP request to a third party (Strava, Wahoo) and per outbound
    email. Its volume profile is an order of magnitude away from the LLM table's
    — machine-driven bursts during a backfill against one row per deliberate
    user action — so it gets its own file and its own retention policy, and
    ``llm_usage.db`` keeps its accurate name.
    """
    pass


def _set_wal_mode(dbapi_conn, _connection_record) -> None:
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()
