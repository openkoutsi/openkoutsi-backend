"""LLM subscription gating: entitlements table + instance opt-in switch (issue #9).

Adds:

* ``instance_settings.llm_requires_subscription`` — boolean, non-null, default
  false. The opt-in gate; until an admin flips it, LLM features work as today.
* ``llm_entitlements`` — one row per user granting "LLM access". A table (not a
  role) so it can carry expiry, provenance and audit fields, and act as an
  idempotent upsert target for the future payment handler (#16).

The per-call token usage lives in a **separate** database (its own Alembic
chain), not here.

Revision ID: 010_llm_entitlements
Revises: 009_instance_llm_default_first_preset
Create Date: 2026-07-09
"""
import sqlalchemy as sa
from alembic import op

revision = "010_llm_entitlements"
down_revision = "009_instance_llm_default_first_preset"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "instance_settings",
        sa.Column(
            "llm_requires_subscription",
            sa.Boolean(),
            nullable=False,
            server_default="0",
        ),
    )

    op.create_table(
        "llm_entitlements",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("granted_by_user_id", sa.String(), nullable=True),
        sa.Column("starts_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("external_ref", sa.String(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["granted_by_user_id"], ["users.id"], ondelete="SET NULL"
        ),
        sa.UniqueConstraint("user_id", name="uq_llm_entitlements_user_id"),
    )


def downgrade() -> None:
    op.drop_table("llm_entitlements")
    with op.batch_alter_table("instance_settings") as batch_op:
        batch_op.drop_column("llm_requires_subscription")
