"""Add join_requests table for self-serve team join requests.

Revision ID: 005_join_requests
Revises: 004_team_llm_context
Create Date: 2026-06-11
"""
import sqlalchemy as sa
from alembic import op

revision = "005_join_requests"
down_revision = "004_team_llm_context"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "join_requests",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("team_id", sa.String(), nullable=False),
        sa.Column("username", sa.String(), nullable=False),
        sa.Column("password_hash", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decided_by_user_id", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["team_id"], ["teams.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("join_requests")
