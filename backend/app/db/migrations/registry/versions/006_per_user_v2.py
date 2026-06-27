"""v2: collapse the team layer into a single instance with per-user data.

Adds the global ``roles`` + consent columns to ``users`` and the single-row
``instance_settings`` table, makes invitations instance-wide (drops ``team_id``),
and drops the team tables.

IMPORTANT: run the one-time data migration script
``backend/scripts/migrate_to_per_user.py`` *before* applying this revision — it
reads ``teams`` / ``team_memberships`` / ``data_consents`` which this revision
removes.

Revision ID: 006_per_user_v2
Revises: 005_join_requests
Create Date: 2026-06-27
"""
import sqlalchemy as sa
from alembic import op

revision = "006_per_user_v2"
down_revision = "005_join_requests"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # users: global roles + consent (absorbs data_consents)
    op.add_column(
        "users",
        sa.Column("roles", sa.String(), nullable=False, server_default='["user"]'),
    )
    op.add_column("users", sa.Column("consented_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column("consent_version", sa.String(), nullable=True))

    # instance-wide settings (replaces per-team LLM overrides)
    op.create_table(
        "instance_settings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("llm_base_url", sa.String(), nullable=True),
        sa.Column("llm_api_key_enc", sa.String(), nullable=True),
        sa.Column("llm_model", sa.String(), nullable=True),
        sa.Column("llm_analysis_context", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # invitations become instance-wide
    with op.batch_alter_table("invitations") as batch:
        batch.drop_column("team_id")

    # drop the team layer
    for table in ("join_requests", "data_consents", "team_memberships", "teams"):
        op.drop_table(table)


def downgrade() -> None:
    raise NotImplementedError("Downgrade from the per-user v2 schema is not supported.")
