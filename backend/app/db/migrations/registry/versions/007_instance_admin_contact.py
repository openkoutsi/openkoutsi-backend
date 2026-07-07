"""Add admin_contact column to instance_settings table.

Revision ID: 007_instance_admin_contact
Revises: 006_per_user_v2
Create Date: 2026-07-07
"""
import sqlalchemy as sa
from alembic import op

revision = "007_instance_admin_contact"
down_revision = "006_per_user_v2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("instance_settings", sa.Column("admin_contact", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("instance_settings", "admin_contact")
