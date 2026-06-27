"""Add labels and notes columns to activities table.

Revision ID: 002_activity_labels_notes
Revises: 001_workout_definitions
Create Date: 2026-05-26

Idempotent: safe to run against DBs that already have these columns.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "002_activity_labels_notes"
down_revision = "001_workout_definitions"
branch_labels = None
depends_on = None


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
    return any(row[1] == column_name for row in rows)


def upgrade() -> None:
    conn = op.get_bind()
    if not _column_exists(conn, "activities", "labels"):
        op.add_column("activities", sa.Column("labels", sa.JSON(), nullable=True))
    if not _column_exists(conn, "activities", "notes"):
        op.add_column("activities", sa.Column("notes", sa.Text(), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    if _column_exists(conn, "activities", "notes"):
        op.drop_column("activities", "notes")
    if _column_exists(conn, "activities", "labels"):
        op.drop_column("activities", "labels")
