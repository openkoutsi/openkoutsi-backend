"""Add ``courses.target_power_w`` (issue #61).

A course was pace-able to a finish time or to nothing at all; the other half of
the question — "what does this look like if I just sit on 210 W?" — had nowhere
to live.

The two targets are alternatives, not a pair: the API rejects a request carrying
both and clears one when the other is set. Enforced there rather than as a CHECK
constraint, because SQLite cannot add one to an existing table without a rebuild,
which is a bigger risk than the rule is worth.

Nullable and un-backfilled: every existing course was solved for a time or for
nothing, and neither is a power target.

Idempotent, like every migration in this tree.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "026_course_target_power"
down_revision = "025_courses_bikes"
branch_labels = None
depends_on = None

_TABLE = "courses"
_COLUMN = "target_power_w"


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
    if _table_exists(conn, _TABLE) and not _column_exists(conn, _TABLE, _COLUMN):
        op.add_column(_TABLE, sa.Column(_COLUMN, sa.Integer(), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    if _table_exists(conn, _TABLE) and _column_exists(conn, _TABLE, _COLUMN):
        op.drop_column(_TABLE, _COLUMN)
