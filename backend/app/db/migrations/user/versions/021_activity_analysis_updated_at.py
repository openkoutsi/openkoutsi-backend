"""Add ``activities.analysis_updated_at`` (issue #91).

Training status and goal guidance both carry a timestamp their routers age a
stuck ``pending`` against; the activity analysis had the status and the prose but
no clock. Since ``trigger_analysis`` early-returns for ``pending``, a row whose
process died mid-run left that activity permanently un-analysable.

Nullable and un-backfilled: a pre-existing ``pending`` row has no known
last-progress time, and the age check treats NULL as immediately timed out.

Idempotent, like every migration in this tree.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "021_activity_analysis_updated_at"
down_revision = "020_agent_progress"
branch_labels = None
depends_on = None

_TABLE = "activities"
_COLUMN = "analysis_updated_at"


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
