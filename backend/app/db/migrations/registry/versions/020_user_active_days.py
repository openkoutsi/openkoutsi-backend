"""Daily active users.

Adds ``user_active_days``: one row per (user, UTC day) on which the user loaded
the dashboard. The admin ``GET /api/admin/active-users/summary`` endpoint counts
distinct users per day, week or month from it.

Nothing is backfilled — there is no record of past use to backfill from — so
the counts start on the day an instance upgrades.

Revision ID: 020_user_active_days
Revises: 019_signup_halt
Create Date: 2026-11-08
"""
import sqlalchemy as sa
from alembic import op

revision = "020_user_active_days"
down_revision = "019_signup_halt"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_active_days",
        sa.Column(
            "user_id",
            sa.String(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("day", sa.Date(), primary_key=True),
    )
    op.create_index("ix_user_active_days_day", "user_active_days", ["day"])


def downgrade() -> None:
    op.drop_index("ix_user_active_days_day", table_name="user_active_days")
    op.drop_table("user_active_days")
