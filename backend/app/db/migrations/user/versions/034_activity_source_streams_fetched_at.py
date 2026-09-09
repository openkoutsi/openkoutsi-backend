"""Record when a source's detail data was last resolved (issue #67).

The sync loop skipped an activity whenever an ``ActivitySource`` existed for its
(provider, external_id) pair, which conflated "we imported this" with "we
imported this *completely*". Behind a 429 the two came apart: the activity was
created, its streams were not, and the skip meant no later sync ever looked at
it again. The athlete's fitness and fatigue were then computed from a ride with
no power or HR behind it, with nothing anywhere saying so.

``streams_fetched_at`` is what the skip now consults. NULL means this source's
detail data has never been resolved against the provider and the next sync
should try again.

**The backfill matters as much as the column.** Defaulting every existing row to
NULL would send the first sync after this deploys back to the provider for the
athlete's entire history — the exact quota burst the issue is about. So rows
that plainly did land are stamped as fetched:

  * the source has a stored file (``fit_file_path``), or
  * its activity has at least one stream row.

What is left NULL is what is actually suspect: a source with no file whose
activity has no streams either. Those get one repair attempt on the next sync
and are then settled either way, which is also how already-affected athletes are
healed rather than only new imports being protected.

The stamp is ``created_at`` rather than now: it is when the data was fetched,
and dating it to the migration would claim a fetch that never happened today.

Idempotent, like every migration in this tree.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "034_activity_source_streams_fetched_at"
down_revision = "033_plan_completed_at"
branch_labels = None
depends_on = None


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
    return any(row[1] == column_name for row in rows)


def upgrade() -> None:
    conn = op.get_bind()
    if _column_exists(conn, "activity_sources", "streams_fetched_at"):
        return

    op.add_column(
        "activity_sources",
        sa.Column("streams_fetched_at", sa.DateTime(timezone=True), nullable=True),
    )
    conn.execute(
        text(
            """
            UPDATE activity_sources
               SET streams_fetched_at = created_at
             WHERE fit_file_path IS NOT NULL
                OR EXISTS (
                        SELECT 1
                          FROM activity_streams
                         WHERE activity_streams.activity_id
                               = activity_sources.activity_id
                   )
            """
        )
    )


def downgrade() -> None:
    conn = op.get_bind()
    if _column_exists(conn, "activity_sources", "streams_fetched_at"):
        op.drop_column("activity_sources", "streams_fetched_at")
