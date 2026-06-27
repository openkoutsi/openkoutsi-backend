"""Add workout_category column to activities table.

Revision ID: 003_workout_category
Revises: 002_activity_intervals
Create Date: 2026-04-30
"""
from alembic import op
import sqlalchemy as sa

revision = "003_workout_category"
down_revision = "002_activity_intervals"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "activities",
        sa.Column("workout_category", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("activities", "workout_category")
