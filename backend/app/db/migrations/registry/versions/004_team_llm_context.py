"""Add llm_analysis_context column to teams table.

Revision ID: 004_team_llm_context
Revises: 003_data_consents
Create Date: 2026-05-17
"""
import sqlalchemy as sa
from alembic import op

revision = "004_team_llm_context"
down_revision = "003_data_consents"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("teams", sa.Column("llm_analysis_context", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("teams", "llm_analysis_context")
