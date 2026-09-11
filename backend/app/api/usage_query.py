"""Shared query helpers for the usage-summary endpoints (issues #9, #66).

Lifted out of :mod:`backend.app.api.admin` when the third-party API-usage
summary arrived so the two summaries share one definition of a time bucket
instead of drifting apart. Both are SQL-expression-level and DB-agnostic, so the
fact that the two tables live in different SQLite files does not affect them —
and SQLite's ``%W`` week semantics stay identical across both, which is what
keeps a "week" the same week in either table.
"""

from datetime import datetime

from fastapi import HTTPException
from sqlalchemy import func

#: SQLite strftime formats for the time-bucketed ``group_by`` values.
TIME_BUCKETS = {
    "day": "%Y-%m-%d",
    "week": "%Y-W%W",
    "month": "%Y-%m",
}


def parse_usage_dt(raw: str | None):
    """Parse a ``from``/``to`` query parameter, or 400 on nonsense."""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date/time '{raw}' — use ISO format (YYYY-MM-DD).",
        )


def time_bucket_expr(group_by: str, column):
    """The ``strftime`` expression bucketing *column* by *group_by*."""
    return func.strftime(TIME_BUCKETS[group_by], column)
