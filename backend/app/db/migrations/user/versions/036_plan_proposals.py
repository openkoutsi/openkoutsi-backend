"""Add plan_proposals (issue #72).

Koutsi can now *draft* a training plan — a new one, or a change to an existing
one — and put it in front of the athlete as a structured yes/no. The draft has
to live somewhere between being offered and being decided, and that somewhere is
this table.

Not a ``training_plans`` row with a status, which is the tempting shape and the
wrong one: a plan row is picked up by the plan page, the adherence snapshots, the
activity matcher and the achievements, so a "draft" plan would have to be
excluded from each of them separately, and the first place that forgot would be
a plan the athlete never agreed to. A separate table is inert by construction.

``payload`` holds the *resolved* change — the weeks as built — rather than the
model's arguments, so approving replays nothing through a second completion.
``summary`` holds the preview the card and the model both read, including the
plans an approval would archive.

No backfill: nothing has ever proposed anything.

Idempotent, like every migration in this tree.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "036_plan_proposals"
down_revision = "035_provider_sync_state"
branch_labels = None
depends_on = None


def _table_exists(conn, table_name: str) -> bool:
    row = conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='table' AND name=:n"),
        {"n": table_name},
    ).fetchone()
    return row is not None


def _index_exists(conn, index_name: str) -> bool:
    row = conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='index' AND name=:n"),
        {"n": index_name},
    ).fetchone()
    return row is not None


def upgrade() -> None:
    conn = op.get_bind()

    if not _table_exists(conn, "plan_proposals"):
        op.create_table(
            "plan_proposals",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("conversation_id", sa.String(), nullable=True),
            sa.Column("message_id", sa.String(), nullable=True),
            sa.Column("kind", sa.String(), nullable=False),
            sa.Column("target_plan_id", sa.String(), nullable=True),
            sa.Column("target_workout_id", sa.String(), nullable=True),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("summary", sa.JSON(), nullable=False),
            sa.Column("status", sa.String(), nullable=False, server_default="pending"),
            sa.Column("built_by", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("applied_plan_id", sa.String(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _index_exists(conn, "ix_plan_proposals_conversation"):
        op.create_index(
            "ix_plan_proposals_conversation",
            "plan_proposals",
            ["conversation_id", "status"],
        )
    if not _index_exists(conn, "ix_plan_proposals_message"):
        op.create_index(
            "ix_plan_proposals_message", "plan_proposals", ["message_id"]
        )


def downgrade() -> None:
    conn = op.get_bind()
    if _index_exists(conn, "ix_plan_proposals_message"):
        op.drop_index("ix_plan_proposals_message", table_name="plan_proposals")
    if _index_exists(conn, "ix_plan_proposals_conversation"):
        op.drop_index("ix_plan_proposals_conversation", table_name="plan_proposals")
    if _table_exists(conn, "plan_proposals"):
        op.drop_table("plan_proposals")
