"""Add data_consents table for user privacy acceptance tracking.

Revision ID: 003_data_consents
Revises: 002_team_status
Create Date: 2026-05-16
"""
import sqlalchemy as sa
from alembic import op

revision = "003_data_consents"
down_revision = "002_team_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "data_consents",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("team_id", sa.String(), nullable=False),
        sa.Column("consented_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consent_version", sa.String(), nullable=False, server_default="1.0"),
        sa.ForeignKeyConstraint(["team_id"], ["teams.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "team_id"),
    )


def downgrade() -> None:
    op.drop_table("data_consents")
