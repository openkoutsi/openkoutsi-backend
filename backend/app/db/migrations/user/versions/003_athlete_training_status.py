"""Add training_status columns to athletes table.

Revision ID: 003_athlete_training_status
Revises: 002_activity_labels_notes
Create Date: 2026-05-26

Idempotent: safe to run against DBs that already have these columns.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "003_athlete_training_status"
down_revision = "002_activity_labels_notes"
branch_labels = None
depends_on = None


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
    return any(row[1] == column_name for row in rows)


def upgrade() -> None:
    conn = op.get_bind()
    if not _column_exists(conn, "athletes", "training_status"):
        op.add_column("athletes", sa.Column("training_status", sa.Text(), nullable=True))
    if not _column_exists(conn, "athletes", "training_status_status"):
        op.add_column("athletes", sa.Column("training_status_status", sa.String(), nullable=True))
    if not _column_exists(conn, "athletes", "training_status_date"):
        op.add_column("athletes", sa.Column("training_status_date", sa.Date(), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    if _column_exists(conn, "athletes", "training_status_date"):
        op.drop_column("athletes", "training_status_date")
    if _column_exists(conn, "athletes", "training_status_status"):
        op.drop_column("athletes", "training_status_status")
    if _column_exists(conn, "athletes", "training_status"):
        op.drop_column("athletes", "training_status")
