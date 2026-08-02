"""Drop activity links that point at a rest day (issue #40).

A plan carries a ``planned_workouts`` row for every day of the week, rest days
included. The auto-matcher used to treat those rows as ordinary candidates, and
a rest day is the loosest target a plan can offer: ``sports_match`` accepts
"rest" for any endurance sport, and its NULL ``target_load``/``duration_min``
make both threshold gates pass unconditionally. Any session ridden on a rest day
was therefore linked to it on ingest.

Nothing surfaced the link — the dashboard calendar skips rest rows, adherence
scoring skips rest rows, and the activity itself shows no plan linkage — but it
still occupied the activity's one allowed link, so linking that activity to the
session it actually completed failed with "already linked to another planned
workout" and nothing visible to unlink.

The matcher no longer produces these; this clears the ones already written.
Deleting them changes no adherence score or achievement: ``score_plan`` excludes
rest days from both the numerator and the denominator, so these links never
contributed anything.

Scope is deliberately narrower than ``is_rest_workout``. That helper also treats
an *untyped* row as a rest day, but an untyped row can carry a real prescription
— the LLM generator passes the model's ``workout_type`` through verbatim, and
``PlannedWorkoutUpdate`` accepts ``""`` — and the manual link endpoint has never
refused a rest day, so such a link may have been made on purpose. An untyped row
is therefore only cleared when it prescribes nothing, which is what a genuine
``plan_builder._rest_day`` looks like. The matcher declining to *create* a link
is cheap to be liberal about; deleting one already in the database is not.

Every deleted row is copied to ``planned_workout_activities_dropped_019`` first
and the count is logged, so a mistake in the predicate is recoverable by hand
rather than only discoverable from a user report.

Idempotent: a plain DELETE over a predicate, safe to re-run and safe on a DB
built fresh by SQLAlchemy create_all (which has the table but no rows to clear).
"""
import logging

from alembic import op
from sqlalchemy import text

revision = "019_drop_rest_day_activity_links"
down_revision = "018_activity_aerobic_metrics"
branch_labels = None
depends_on = None

log = logging.getLogger("alembic.runtime.migration")

BACKUP_TABLE = "planned_workout_activities_dropped_019"

# Rows whose planned workout is a rest day, qualified with an alias so the
# columns cannot silently bind to the outer table — an unqualified column that
# does not exist on the inner table resolves against the outer one in SQLite and
# collapses the predicate to TRUE, deleting everything.
#
# SQLite's TRIM strips spaces only, where Python's str.strip() strips all
# whitespace; the explicit character set keeps this in step with
# ``openkoutsi.sport_matching.is_rest_workout``.
_REST_LINK_PREDICATE = """
    planned_workout_id IN (
        SELECT pw.id FROM planned_workouts pw
        WHERE TRIM(LOWER(pw.workout_type), ' ' || char(9) || char(10) || char(13)) = 'rest'
           OR (
                (pw.workout_type IS NULL
                 OR TRIM(pw.workout_type, ' ' || char(9) || char(10) || char(13)) = '')
                AND pw.target_load IS NULL
                AND pw.duration_min IS NULL
           )
    )
"""


def _table_exists(conn, table_name: str) -> bool:
    row = conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='table' AND name=:n"),
        {"n": table_name},
    ).fetchone()
    return row is not None


def upgrade() -> None:
    conn = op.get_bind()

    if not _table_exists(conn, "planned_workout_activities"):
        return
    if not _table_exists(conn, "planned_workouts"):
        return

    doomed = conn.execute(
        text(
            "SELECT count(*) FROM planned_workout_activities "
            f"WHERE {_REST_LINK_PREDICATE}"
        )
    ).scalar_one()

    if not doomed:
        log.info("019: no rest-day activity links to drop")
        return

    # Snapshot before deleting. CREATE TABLE … AS on the first run, INSERT on a
    # re-run, so re-applying never discards an earlier snapshot.
    if _table_exists(conn, BACKUP_TABLE):
        conn.execute(
            text(
                f"INSERT INTO {BACKUP_TABLE} "
                "SELECT * FROM planned_workout_activities "
                f"WHERE {_REST_LINK_PREDICATE}"
            )
        )
    else:
        conn.execute(
            text(
                f"CREATE TABLE {BACKUP_TABLE} AS "
                "SELECT * FROM planned_workout_activities "
                f"WHERE {_REST_LINK_PREDICATE}"
            )
        )

    conn.execute(
        text(f"DELETE FROM planned_workout_activities WHERE {_REST_LINK_PREDICATE}")
    )
    log.info(
        "019: dropped %d rest-day activity link(s); copies kept in %s",
        doomed,
        BACKUP_TABLE,
    )


def downgrade() -> None:
    """Deliberately restores nothing.

    The rows this migration removes are spurious auto-links: putting them back
    would put the bug back with them, so the downgrade declines rather than
    being unable. They are not lost — ``planned_workout_activities_dropped_019``
    holds a copy of every deleted row, so a restore is a single INSERT … SELECT
    away for an operator who decides they want one.
    """
