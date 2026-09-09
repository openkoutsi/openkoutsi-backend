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

What is left NULL is what is actually suspect: a source with no file that cannot
be shown to own the streams on its activity. Those get one repair attempt on the
next sync and are then settled either way, which is also how already-affected
athletes are healed rather than only new imports being protected.

**Why the streams half only counts single-source activities.** ``EXISTS`` on
``activity_streams`` is keyed on the *activity*, and an activity can have several
sources. On a two-source ride, a Wahoo source whose FIT download was throttled
owns no file and resolved nothing — but the activity has streams, because Strava
put them there, so an unqualified ``EXISTS`` would stamp it. Those multi-source
rides are the likeliest shape of the already-hollow history this backfill is
meant to leave repairable, so they are excluded: only a source that is the sole
one on its activity can be credited with that activity's streams. Single-source
rides are the overwhelming majority and the whole quota-burst concern, and they
stay stamped.

The stamp is ``created_at`` rather than now: it is when the data was fetched,
and dating it to the migration would claim a fetch that never happened today.

**The index is not incidental.** The correlated subquery below has no index to
use on ``activity_streams.activity_id``, so it re-scans the largest table in the
database once per source row — measured at 17 s on a 4 000-activity history, and
the cost is the product of the two tables. Creating the index first takes that to
well under a second, and it earns its keep afterwards: every stream delete in
``_repopulate_activity`` is the same lookup. It is declared on the model now, so
``downgrade`` leaves it alone — dropping it would put a fresh database and a
migrated one out of step.

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


def _index_exists(conn, index_name: str) -> bool:
    row = conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='index' AND name=:n"),
        {"n": index_name},
    ).fetchone()
    return row is not None


def upgrade() -> None:
    conn = op.get_bind()

    # First, and regardless of the column: the backfill below reads this table
    # once per source row, and every stream delete in the sync does the same
    # lookup. Separate from the column check so a database that somehow has one
    # and not the other still ends up with both.
    if not _index_exists(conn, "ix_activity_streams_activity_id"):
        op.create_index(
            "ix_activity_streams_activity_id", "activity_streams", ["activity_id"]
        )

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
                OR (
                        NOT EXISTS (
                            SELECT 1
                              FROM activity_sources AS sibling
                             WHERE sibling.activity_id
                                   = activity_sources.activity_id
                               AND sibling.id <> activity_sources.id
                        )
                    AND EXISTS (
                            SELECT 1
                              FROM activity_streams
                             WHERE activity_streams.activity_id
                                   = activity_sources.activity_id
                        )
                   )
            """
        )
    )


def downgrade() -> None:
    conn = op.get_bind()
    if _column_exists(conn, "activity_sources", "streams_fetched_at"):
        op.drop_column("activity_sources", "streams_fetched_at")
