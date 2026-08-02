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

Idempotent: a plain DELETE over a predicate, safe to re-run and safe on a DB
built fresh by SQLAlchemy create_all (which has the table but no rows to clear).
"""
from alembic import op
from sqlalchemy import text

revision = "019_drop_rest_day_activity_links"
down_revision = "018_activity_aerobic_metrics"
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

    if not _table_exists(conn, "planned_workout_activities"):
        return
    if not _table_exists(conn, "planned_workouts"):
        return

    # Mirrors ``openkoutsi.sport_matching.is_rest_workout``: an untyped row is a
    # placeholder of the same kind as an explicit rest day.
    conn.execute(
        text(
            """
            DELETE FROM planned_workout_activities
            WHERE planned_workout_id IN (
                SELECT id FROM planned_workouts
                WHERE workout_type IS NULL
                   OR TRIM(LOWER(workout_type)) IN ('', 'rest')
            )
            """
        )
    )


def downgrade() -> None:
    """No-op.

    The rows this migration removes were spurious auto-links, and nothing records
    which activity had been attached to which rest day. Re-creating them would
    mean re-creating the bug, so the downgrade deliberately restores nothing.
    """
