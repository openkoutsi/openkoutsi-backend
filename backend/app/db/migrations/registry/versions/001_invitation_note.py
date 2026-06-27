"""Add note column to invitations table.

Revision ID: 001_invitation_note
Revises:
Create Date: 2026-05-02
"""
import sqlalchemy as sa
from alembic import op

revision = "001_invitation_note"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "invitations",
        sa.Column("note", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("invitations", "note")
