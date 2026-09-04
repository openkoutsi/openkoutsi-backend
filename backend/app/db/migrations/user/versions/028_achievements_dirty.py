"""Add ``athletes.achievements_dirty_at`` (issue #69).

``recompute_achievements`` re-reads the athlete's entire activity history plus
every plan on every call, and used to run inline on every ingest event — so
importing a season was N events × O(N) activities. A write path now stamps this
column and returns, and the next achievements read or the daily sweep does the
one reconcile that settles them all.

Nullable and un-backfilled: NULL means "no recompute is owed", correct for every
existing athlete because the code this replaces recomputed eagerly.

Idempotent, like every migration in this tree.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "028_achievements_dirty"
down_revision = "027_llm_run_tokens"
branch_labels = None
depends_on = None

_TABLE = "athletes"
_COLUMN = "achievements_dirty_at"


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
    return any(row[1] == column_name for row in rows)


def upgrade() -> None:
    conn = op.get_bind()
    if not _column_exists(conn, _TABLE, _COLUMN):
        op.add_column(
            _TABLE, sa.Column(_COLUMN, sa.DateTime(timezone=True), nullable=True)
        )


def downgrade() -> None:
    conn = op.get_bind()
    if _column_exists(conn, _TABLE, _COLUMN):
        op.drop_column(_TABLE, _COLUMN)
