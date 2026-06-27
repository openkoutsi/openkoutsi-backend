"""Add skip_reason column to planned_workouts table.

Revision ID: 005_planned_workout_skip_reason
Revises: 004_goal_outcome_note
Create Date: 2026-06-06

Idempotent: safe to run against DBs that already have this column.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "005_planned_workout_skip_reason"
down_revision = "004_goal_outcome_note"
branch_labels = None
depends_on = None


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
    return any(row[1] == column_name for row in rows)


def upgrade() -> None:
    conn = op.get_bind()
    if not _column_exists(conn, "planned_workouts", "skip_reason"):
        op.add_column("planned_workouts", sa.Column("skip_reason", sa.String(), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    if _column_exists(conn, "planned_workouts", "skip_reason"):
        op.drop_column("planned_workouts", "skip_reason")
