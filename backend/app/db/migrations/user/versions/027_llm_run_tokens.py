"""Add run tokens to the three LLM surfaces that lacked one (issue #50).

``courses.plan_run_id`` already did this: it identifies the run that owns the
columns beside it, so a run whose token no longer matches discards its own
writes rather than putting a stale answer back on the row. The training status,
goal guidance and activity analysis had no equivalent — and a ``pending`` row
that a read has settled is re-triggerable while its original run may still be
alive and merely slow.

Nullable and un-backfilled: a pre-existing row carries ``NULL``, which
``run_is_current`` reads as the old behaviour, so nothing in flight across the
upgrade is discarded.

Idempotent, like every migration here — safe against DBs that already have the
columns, including ones built by ``create_all`` with no alembic stamp.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "027_llm_run_tokens"
down_revision = "026_course_target_power"
branch_labels = None
depends_on = None

#: (table, column) — one per surface that writes a `pending` LLM status.
_COLUMNS = [
    ("athletes", "training_status_run_id"),
    ("goals", "guidance_run_id"),
    ("activities", "analysis_run_id"),
]


def _table_exists(conn, table_name: str) -> bool:
    row = conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='table' AND name=:n"),
        {"n": table_name},
    ).fetchone()
    return row is not None


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
    return any(row[1] == column_name for row in rows)


def upgrade() -> None:
    conn = op.get_bind()
    for table, column in _COLUMNS:
        if _table_exists(conn, table) and not _column_exists(conn, table, column):
            op.add_column(table, sa.Column(column, sa.String(), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    for table, column in _COLUMNS:
        if _table_exists(conn, table) and _column_exists(conn, table, column):
            op.drop_column(table, column)
