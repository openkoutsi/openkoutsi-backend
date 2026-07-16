"""Email-based signup + self-serve password reset (issue #15).

Adds:

* ``users.email`` (unique, nullable) + ``users.email_verified_at`` — the email
  login identifier for self-serve signup, verified before the account activates.
* ``users.username`` becomes **nullable** — email-only signup accounts need no
  synthetic username; at least one of email/username is always present.
* ``email_verification_tokens`` — single-use, hashed, expiring tokens mirroring
  ``password_reset_tokens``.
* ``instance_settings.allow_self_signup`` — boolean, non-null, default false. The
  admin opt-in gate; the instance stays invite-only until flipped.

Revision ID: 011_email_signup
Revises: 010_llm_entitlements
Create Date: 2026-07-15
"""
import sqlalchemy as sa
from alembic import op

revision = "011_email_signup"
down_revision = "010_llm_entitlements"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("email", sa.String(), nullable=True))
    op.add_column(
        "users",
        sa.Column("email_verified_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("uq_users_email", "users", ["email"], unique=True)
    # username was NOT NULL; relax it so email-only accounts can omit it.
    with op.batch_alter_table("users") as batch_op:
        batch_op.alter_column(
            "username", existing_type=sa.String(), nullable=True
        )

    op.add_column(
        "instance_settings",
        sa.Column(
            "allow_self_signup",
            sa.Boolean(),
            nullable=False,
            server_default="0",
        ),
    )

    op.create_table(
        "email_verification_tokens",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("token_hash", sa.String(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("token_hash", name="uq_email_verification_tokens_token_hash"),
    )


def downgrade() -> None:
    op.drop_table("email_verification_tokens")
    with op.batch_alter_table("instance_settings") as batch_op:
        batch_op.drop_column("allow_self_signup")
    with op.batch_alter_table("users") as batch_op:
        batch_op.alter_column(
            "username", existing_type=sa.String(), nullable=False
        )
    op.drop_index("uq_users_email", table_name="users")
    op.drop_column("users", "email_verified_at")
    op.drop_column("users", "email")
