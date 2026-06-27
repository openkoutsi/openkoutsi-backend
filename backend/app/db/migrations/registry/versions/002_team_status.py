"""Add status column to teams table.

Revision ID: 002_team_status
Revises: 001_invitation_note
Create Date: 2026-05-02
"""
import sqlalchemy as sa
from alembic import op

revision = "002_team_status"
down_revision = "001_invitation_note"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "teams",
        sa.Column("status", sa.String(), nullable=False, server_default="active"),
    )


def downgrade() -> None:
    op.drop_column("teams", "status")
