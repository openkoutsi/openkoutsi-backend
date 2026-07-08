"""Add llm_models and llm_extra_headers columns to instance_settings.

Adds support for configuring several selectable models (each with its own
extra chat-completion body params) and arbitrary extra HTTP headers applied to
every outbound LLM request.

Revision ID: 008_instance_llm_models_headers
Revises: 007_instance_admin_contact
Create Date: 2026-07-08
"""
import sqlalchemy as sa
from alembic import op

revision = "008_instance_llm_models_headers"
down_revision = "007_instance_admin_contact"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("instance_settings", sa.Column("llm_models", sa.JSON(), nullable=True))
    op.add_column("instance_settings", sa.Column("llm_extra_headers", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("instance_settings", "llm_extra_headers")
    op.drop_column("instance_settings", "llm_models")
