"""Add AI-guidance columns to the goals table.

Revision ID: 008_goal_guidance
Revises: 007_wahoo_workout_uploads
Create Date: 2026-07-13

Idempotent: safe to run against DBs that already have these columns.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "008_goal_guidance"
down_revision = "007_wahoo_workout_uploads"
branch_labels = None
depends_on = None


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
    return any(row[1] == column_name for row in rows)


def upgrade() -> None:
    conn = op.get_bind()
    if not _column_exists(conn, "goals", "guidance"):
        op.add_column("goals", sa.Column("guidance", sa.Text(), nullable=True))
    if not _column_exists(conn, "goals", "guidance_verdict"):
        op.add_column("goals", sa.Column("guidance_verdict", sa.String(), nullable=True))
    if not _column_exists(conn, "goals", "guidance_status"):
        op.add_column("goals", sa.Column("guidance_status", sa.String(), nullable=True))
    if not _column_exists(conn, "goals", "guidance_updated_at"):
        op.add_column(
            "goals",
            sa.Column("guidance_updated_at", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    conn = op.get_bind()
    for column in ("guidance_updated_at", "guidance_status", "guidance_verdict", "guidance"):
        if _column_exists(conn, "goals", column):
            op.drop_column("goals", column)
