"""Add activity_intervals table for per-interval stats extracted from FIT lap frames.

Revision ID: 002_activity_intervals
Revises: 001_activity_sources
Create Date: 2026-04-29
"""
from alembic import op
import sqlalchemy as sa

revision = "002_activity_intervals"
down_revision = "001_activity_sources"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "activity_intervals",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("activity_id", sa.String(), nullable=False),
        sa.Column("interval_number", sa.Integer(), nullable=False),
        sa.Column("start_offset_s", sa.Integer(), nullable=False),
        sa.Column("duration_s", sa.Integer(), nullable=False),
        sa.Column("distance_m", sa.Float(), nullable=True),
        sa.Column("avg_hr", sa.Float(), nullable=True),
        sa.Column("avg_power", sa.Float(), nullable=True),
        sa.Column("avg_speed_ms", sa.Float(), nullable=True),
        sa.Column("avg_cadence", sa.Float(), nullable=True),
        sa.Column("is_auto_split", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.ForeignKeyConstraint(["activity_id"], ["activities.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_activity_intervals_activity_id",
        "activity_intervals",
        ["activity_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_activity_intervals_activity_id", table_name="activity_intervals")
    op.drop_table("activity_intervals")
