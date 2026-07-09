"""Initial LLM-usage table (issue #9).

One append-only row per instance-paid LLM call. BYOK calls are never recorded,
so there is no ``byok`` column — every row is instance-paid. Input and output
tokens are stored separately (``prompt_tokens`` / ``completion_tokens``); the
``provider`` records which provider served the call alongside ``model``.

Revision ID: 001_llm_usage
Revises:
Create Date: 2026-07-09
"""
import sqlalchemy as sa
from alembic import op

revision = "001_llm_usage"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "llm_usage",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=True),
        sa.Column("model", sa.String(), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("key_source", sa.String(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
    )
    op.create_index("ix_llm_usage_user_id", "llm_usage", ["user_id"])
    op.create_index("ix_llm_usage_created_at", "llm_usage", ["created_at"])
    op.create_index(
        "ix_llm_usage_user_created", "llm_usage", ["user_id", "created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_llm_usage_user_created", table_name="llm_usage")
    op.drop_index("ix_llm_usage_created_at", table_name="llm_usage")
    op.drop_index("ix_llm_usage_user_id", table_name="llm_usage")
    op.drop_table("llm_usage")
