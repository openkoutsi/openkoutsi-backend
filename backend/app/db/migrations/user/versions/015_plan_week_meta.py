"""Add week_meta column to the training_plans table (issue #29).

Stores per-week plan metadata (build vs recovery week, focus note, target weekly
Load/hours, base Load) as a JSON list, populated at generation time by both the
rule-based and LLM plan generators. Nullable — older plans return null until
regenerated.

Idempotent: safe to run against DBs that already have the column (including ones
built fresh by SQLAlchemy create_all, which adds the column but doesn't stamp
alembic).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "015_plan_week_meta"
down_revision = "014_activity_rpe"
branch_labels = None
depends_on = None


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
    return any(row[1] == column_name for row in rows)


def upgrade() -> None:
    conn = op.get_bind()
    if not _column_exists(conn, "training_plans", "week_meta"):
        op.add_column("training_plans", sa.Column("week_meta", sa.JSON(), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    if _column_exists(conn, "training_plans", "week_meta"):
        op.drop_column("training_plans", "week_meta")
