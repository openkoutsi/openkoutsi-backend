"""Add title, body and locale columns to the messages table.

Inbox messages now carry their own text, rendered by
`backend.app.services.message_text` when the message is written, instead of
being reconstructed in the web app from `type` + `data` via an i18n template.
`locale` records which language the text was rendered in so translated
rendering can be added later without another migration.

All three columns are nullable: messages already sitting in mailboxes were
written before the text existed.

Idempotent: safe to run against DBs that already have the columns. That matters
more than usual here — the `messages` table has never had a migration of its
own and exists only because SQLAlchemy `create_all` builds it, so live user DBs
carry the table without being stamped at any particular revision.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "017_message_title_body"
down_revision = "016_achievements"
branch_labels = None
depends_on = None

_COLUMNS = ("title", "body", "locale")


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
    return any(row[1] == column_name for row in rows)


def _table_exists(conn, table_name: str) -> bool:
    row = conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='table' AND name=:n"),
        {"n": table_name},
    ).fetchone()
    return row is not None


def upgrade() -> None:
    conn = op.get_bind()
    if not _table_exists(conn, "messages"):
        return
    for column in _COLUMNS:
        if not _column_exists(conn, "messages", column):
            op.add_column("messages", sa.Column(column, sa.String(), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    if not _table_exists(conn, "messages"):
        return
    for column in _COLUMNS:
        if _column_exists(conn, "messages", column):
            op.drop_column("messages", column)
