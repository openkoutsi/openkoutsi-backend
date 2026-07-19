"""Add rpe column to the activities table (issue #28).

Stores the athlete's subjective Rate of Perceived Exertion (a 1–10 effort
score) for a ride. Nullable until the athlete rates the activity.

Idempotent: safe to run against DBs that already have the column (including ones
built fresh by SQLAlchemy create_all, which adds the column but doesn't stamp
alembic).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "014_activity_rpe"
down_revision = "013_activity_zone_times"
branch_labels = None
depends_on = None


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
    return any(row[1] == column_name for row in rows)


def upgrade() -> None:
    conn = op.get_bind()
    if not _column_exists(conn, "activities", "rpe"):
        op.add_column("activities", sa.Column("rpe", sa.Integer(), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    if _column_exists(conn, "activities", "rpe"):
        op.drop_column("activities", "rpe")
