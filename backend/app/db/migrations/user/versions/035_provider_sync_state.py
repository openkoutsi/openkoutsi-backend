"""Add provider_sync_states (issue #68).

A backfill has four ways to end before it has walked the athlete's history — the
safety limit, a provider that stops serving detail data, a 429, and a lost lease
— and all four returned the same ``(count, earliest)`` tuple as a run that
finished. The only trace was a log line nobody was watching for, so an import
that stopped at 4 000 of 12 000 rides looked exactly like one that imported all
4 000 there were.

This table is where a run now says what it did: whether it finished, which of
the reasons stopped it, how far back it reached, the page the next run should
continue from, and how many runs in a row have ended the same way. That last
column is the one no single run can produce: stopping in the same place three
times is a provider problem or a poisoned range of activities, and it is
invisible unless something remembers across runs.

One row per provider. The database is already per-user, so (user, provider) is
(this file, this primary key).

No backfill: every existing instance has simply never recorded a run, and
inventing one would claim a history this table does not have. The first sync
after this deploys writes the first row.

Idempotent, like every migration in this tree.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "035_provider_sync_state"
down_revision = "034_activity_source_streams_fetched_at"
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

    if not _table_exists(conn, "provider_sync_states"):
        op.create_table(
            "provider_sync_states",
            sa.Column("provider", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("stop_reason", sa.String(), nullable=True),
            sa.Column("stop_detail", sa.String(), nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("imported", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("listed", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("oldest_seen_on", sa.Date(), nullable=True),
            sa.Column("resume_page", sa.Integer(), nullable=True),
            sa.Column("repeat_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("repeat_since", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("provider"),
        )


def downgrade() -> None:
    conn = op.get_bind()
    if _table_exists(conn, "provider_sync_states"):
        op.drop_table("provider_sync_states")
