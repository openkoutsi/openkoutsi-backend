"""Personal access tokens (issue #46).

Adds:

* ``personal_access_tokens`` — long-lived, scoped, revocable credentials a user
  issues to their own tooling. Only ``sha256`` of the secret half is stored;
  expired and revoked rows are retained (the audit log stores token ids, and a
  retained hash keeps a presented-but-revoked token recognisable).
* ``instance_settings.allow_personal_access_tokens`` — boolean, non-null,
  **default true**. The self-hoster's kill switch; unlike the other instance
  gates it defaults on, because a PAT grants strictly less than the session the
  user already holds and there is no prior behaviour to preserve.

Revision ID: 012_personal_access_tokens
Revises: 011_email_signup
Create Date: 2026-08-07
"""
import sqlalchemy as sa
from alembic import op

revision = "012_personal_access_tokens"
down_revision = "011_email_signup"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "instance_settings",
        sa.Column(
            "allow_personal_access_tokens",
            sa.Boolean(),
            nullable=False,
            server_default="1",
        ),
    )

    op.create_table(
        "personal_access_tokens",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("token_hash", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("scopes", sa.String(), nullable=False, server_default="[]"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_expiry_notice", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("token_hash", name="uq_personal_access_tokens_token_hash"),
    )
    op.create_index(
        "ix_personal_access_tokens_user_id",
        "personal_access_tokens",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_personal_access_tokens_user_id", table_name="personal_access_tokens"
    )
    op.drop_table("personal_access_tokens")
    with op.batch_alter_table("instance_settings") as batch_op:
        batch_op.drop_column("allow_personal_access_tokens")
