"""Add chat_conversations / chat_messages (issue #44).

Conversational Koutsi is the first LLM surface with anything to persist: every
other one streams into a column on ``athletes`` and forgets the message list it
built. See ``backend.app.models.chat_orm`` for why tool calls and results are
deliberately not among the things stored.

Idempotent, like every migration in this tree: safe to run against DBs already
migrated or created fresh by SQLAlchemy create_all (which builds the tables but
neither stamps alembic).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "022_chat_conversations"
down_revision = "021_activity_analysis_updated_at"
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

    if not _table_exists(conn, "chat_conversations"):
        op.create_table(
            "chat_conversations",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("title", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _table_exists(conn, "chat_messages"):
        op.create_table(
            "chat_messages",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("conversation_id", sa.String(), nullable=False),
            sa.Column("role", sa.String(), nullable=False),
            sa.Column("content", sa.String(), nullable=False, server_default=""),
            sa.Column("status", sa.String(), nullable=True),
            sa.Column("progress", sa.String(), nullable=True),
            sa.Column("error_code", sa.String(), nullable=True),
            sa.Column("tool_names", sa.JSON(), nullable=True),
            sa.Column("prompt_tokens", sa.Integer(), nullable=True),
            sa.Column("completion_tokens", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(
                ["conversation_id"], ["chat_conversations.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("id"),
        )

    # The thread read is `WHERE conversation_id = ? ORDER BY created_at`, run on
    # every poll — and chat polls sub-second while a turn is live.
    if not _index_exists(conn, "ix_chat_messages_conversation_created"):
        op.create_index(
            "ix_chat_messages_conversation_created",
            "chat_messages",
            ["conversation_id", "created_at"],
        )


def downgrade() -> None:
    conn = op.get_bind()
    if _index_exists(conn, "ix_chat_messages_conversation_created"):
        op.drop_index("ix_chat_messages_conversation_created", table_name="chat_messages")
    if _table_exists(conn, "chat_messages"):
        op.drop_table("chat_messages")
    if _table_exists(conn, "chat_conversations"):
        op.drop_table("chat_conversations")
