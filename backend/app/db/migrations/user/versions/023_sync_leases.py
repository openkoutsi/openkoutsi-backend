"""Add sync_leases (issue #50).

The ±5-minute duplicate check in ``services.provider_sync`` reads the window and
then inserts, and until now the only thing keeping those two steps together was
an ``asyncio.Lock`` — a guarantee about one event loop, offered in place of a
guarantee about the database. Two syncs for one athlete arriving from different
providers (a Wahoo webhook and a Strava backfill, milliseconds apart) is a
routine event, and outside that single loop nothing stopped both from seeing an
empty window and each creating a row for the same ride.

This table is where that guard now lives: one row per named section, claimed with
a conditional UPDATE so the database picks the winner. See
``backend.app.db.leases`` for the mechanics and for why every lease carries a
deadline.

Idempotent, like every migration in this tree: safe to run against DBs already
migrated or created fresh by SQLAlchemy create_all (which builds the tables but
neither stamps alembic).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "023_sync_leases"
down_revision = "022_chat_conversations"
branch_labels = None
depends_on = None


def _table_exists(conn, table_name: str) -> bool:
    row = conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='table' AND name=:n"),
        {"n": table_name},
    ).fetchone()
    return row is not None


def upgrade() -> None:
    conn = op.get_bind()

    if not _table_exists(conn, "sync_leases"):
        op.create_table(
            "sync_leases",
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("holder", sa.String(), nullable=True),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("name"),
        )


def downgrade() -> None:
    conn = op.get_bind()
    if _table_exists(conn, "sync_leases"):
        op.drop_table("sync_leases")
