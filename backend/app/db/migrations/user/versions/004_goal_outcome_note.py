"""Add outcome_note column to goals table.

Revision ID: 004_goal_outcome_note
Revises: 003_athlete_training_status
Create Date: 2026-06-01

Idempotent: safe to run against DBs that already have this column.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "004_goal_outcome_note"
down_revision = "003_athlete_training_status"
branch_labels = None
depends_on = None


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
    return any(row[1] == column_name for row in rows)


def upgrade() -> None:
    conn = op.get_bind()
    if not _column_exists(conn, "goals", "outcome_note"):
        op.add_column("goals", sa.Column("outcome_note", sa.Text(), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    if _column_exists(conn, "goals", "outcome_note"):
        op.drop_column("goals", "outcome_note")
