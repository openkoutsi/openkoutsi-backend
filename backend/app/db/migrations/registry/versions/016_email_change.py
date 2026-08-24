"""Email-address changes, authorised from both ends (issue #62).

Adds ``email_change_tokens``. One row is one pending change and carries **two**
independent secrets — ``new_token_hash``, mailed to the address being claimed,
and ``old_token_hash``, mailed to the address being left. ``users.email`` moves
only once every required side has been stamped, so an unconfirmed request never
touches the login identifier.

Both sides are required because this codebase has no authenticated
change-password endpoint: passwords are set through reset tokens, which are
mailed to ``users.email``. That makes the address the account's only self-serve
root of trust. A one-sided change would let anyone holding just the password
relocate that channel and then take the account permanently via "forgot
password"; requiring the outgoing mailbox costs an attacker exactly what taking
the account over already costs, so the feature adds no new leverage.

``old_token_hash`` is nullable: an invite-created account has no address yet, so
a first set has nothing to authorise against and the new side alone completes
it. ``new_email`` is deliberately **not** unique — two accounts may have a
pending change to the same address at once, nothing is claimed until one of them
finishes, and the unique index on ``users.email`` decides it then.

Revision ID: 016_email_change
Revises: 015_user_token_version
Create Date: 2026-08-24
"""
import sqlalchemy as sa
from alembic import op

revision = "016_email_change"
down_revision = "015_user_token_version"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "email_change_tokens",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("new_token_hash", sa.String(), nullable=False),
        sa.Column("old_token_hash", sa.String(), nullable=True),
        sa.Column("new_email", sa.String(), nullable=False),
        sa.Column("new_confirmed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("old_confirmed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "new_token_hash", name="uq_email_change_tokens_new_token_hash"
        ),
        sa.UniqueConstraint(
            "old_token_hash", name="uq_email_change_tokens_old_token_hash"
        ),
    )
    # Spent rows are retained, so this table only grows, and the live-change
    # lookup runs on every ``GET /auth/account``.
    op.create_index(
        "ix_email_change_tokens_user_id", "email_change_tokens", ["user_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_email_change_tokens_user_id", table_name="email_change_tokens")
    op.drop_table("email_change_tokens")
