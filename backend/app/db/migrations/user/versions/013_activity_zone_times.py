"""Add zone_times snapshot column to the activities table (issue #27).

Stores each activity's accumulated time-in-zone (power + HR), computed at
processing time from the athlete's zones as they were then. Frozen once set so
later zone edits don't rewrite historical weekly zone distributions.

Idempotent: safe to run against DBs that already have the column (including ones
built fresh by SQLAlchemy create_all, which adds the column but doesn't stamp
alembic).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "013_activity_zone_times"
down_revision = "012_plan_adherence_daily"
branch_labels = None
depends_on = None


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
    return any(row[1] == column_name for row in rows)


def upgrade() -> None:
    conn = op.get_bind()
    if not _column_exists(conn, "activities", "zone_times"):
        op.add_column("activities", sa.Column("zone_times", sa.JSON(), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    if _column_exists(conn, "activities", "zone_times"):
        op.drop_column("activities", "zone_times")
