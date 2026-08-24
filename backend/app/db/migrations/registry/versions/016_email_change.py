"""Confirmed email-address changes (issue #62).

Adds ``email_change_tokens`` — single-use, hashed, expiring tokens that each
carry the address they are a claim about (``new_email``). Requesting a change
writes one and mails a link to the *new* address; confirming it is what moves
``users.email``, so an unconfirmed request never touches the login identifier.

A separate table rather than a ``new_email`` column on
``email_verification_tokens``: signup marks every unused verification token a
user holds as spent before issuing a fresh one, and a pending change sharing
that table would be voided by an unrelated signup retry.

``new_email`` is deliberately **not** unique. Two accounts may have a pending
change to the same address simultaneously — nothing is claimed until one of them
confirms, and the unique index on ``users.email`` turns the loser away then.

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
        sa.Column("token_hash", sa.String(), nullable=False),
        sa.Column("new_email", sa.String(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("token_hash", name="uq_email_change_tokens_token_hash"),
    )


def downgrade() -> None:
    op.drop_table("email_change_tokens")
