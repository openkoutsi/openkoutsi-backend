"""Add run tokens to the three LLM surfaces that lacked one (issue #50).

``courses.plan_run_id`` already exists and does this job: it identifies the run
that owns the columns beside it, so a run whose token no longer matches can
discard its own writes instead of putting a stale answer back on the row. The
other three surfaces — the training status, goal guidance and the activity
analysis — had no equivalent, and the failure that leaves open is not
hypothetical.

A ``pending`` row blocks its own re-trigger, so the read paths settle one whose
heartbeat has run down. That is what makes the row re-triggerable, and it is
also what makes the race: the previous run's process may be alive and merely
slow, so it can come back and commit a finished answer over the run the athlete
has just started. The heartbeat says whether a run is *alive*; a token says
whether its writes are still *wanted*. Two questions, two columns.

Nullable and un-backfilled. A row written before this shipped carries ``NULL``,
and ``run_is_current`` reads that as "keeps the old behaviour" — so nothing
in flight across the upgrade is discarded by it.

Idempotent, like every migration in this tree: safe to run against DBs that
already have the columns (including ones built fresh by SQLAlchemy create_all,
which adds them but doesn't stamp alembic).
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
