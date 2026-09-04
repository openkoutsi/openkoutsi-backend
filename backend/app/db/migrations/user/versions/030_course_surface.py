"""Surface classification columns for course recon (issue #56, Stage 2).

Adds nullable columns to three course tables so a matched course can carry the
road surface under it, and how much that surface should be trusted:

* ``course_segments`` — the class, its confidence, exactly what the matcher
  said, and the rolling-resistance coefficient the row was solved with.
* ``courses`` — the background match's status (the ``plan_*`` shape, so
  ``stranded_runs`` settles it at boot), plus ``surface_ribbon``: the surface
  at full run resolution, run-length encoded. The ribbon is separate from the
  segment table on purpose — the table is pacing-shaped and has a minimum row
  length, and a 130 m sector of mud inside 40 km of asphalt has to survive
  into the plan even when the pacing rows fold it into a longer one.
* ``course_tracks`` — the per-point matcher answers, so re-solving for a
  different bike or target costs no re-match.

Adds no rows, deletes none, backfills nothing, and needs no new **mandatory**
environment variables: ``VALHALLA_URL`` is optional and unset by default, and
with it unset a course simply has no surface data — a complete Stage 1 result
rather than a failure. NULL everywhere means "never matched".

Revision ID: 030_course_surface
Revises: 029_activity_label_suggestions
Create Date: 2026-08-31
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy import text

revision = "030_course_surface"
down_revision = "029_activity_label_suggestions"
branch_labels = None
depends_on = None

# (table, column, type). Every one nullable — see the module docstring.
_COLUMNS = [
    ("course_segments", "surface", sa.String()),
    ("course_segments", "surface_confidence", sa.String()),
    ("course_segments", "surface_raw", sa.String()),
    ("course_segments", "crr_used", sa.Float()),
    ("courses", "surface_status", sa.String()),
    ("courses", "surface_run_id", sa.String()),
    ("courses", "surface_updated_at", sa.DateTime(timezone=True)),
    ("courses", "surface_ribbon", sa.JSON()),
    ("course_tracks", "surfaces", sa.JSON()),
    ("course_tracks", "surface_matched_at", sa.DateTime(timezone=True)),
]


def _table_exists(conn, table_name: str) -> bool:
    row = conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='table' AND name=:n"),
        {"n": table_name},
    ).fetchone()
    return row is not None


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
    return any(row[1] == column_name for row in rows)


def upgrade() -> None:
    # Guarded rather than assumed: a database created by `create_all` already
    # has these columns and is stamped at head, but the 001 → head path replays
    # over one that does not, and a database predating course recon has no
    # course tables at all.
    conn = op.get_bind()
    for table, column, type_ in _COLUMNS:
        if _table_exists(conn, table) and not _column_exists(conn, table, column):
            op.add_column(table, sa.Column(column, type_, nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    for table, column, _type in reversed(_COLUMNS):
        if _table_exists(conn, table) and _column_exists(conn, table, column):
            op.drop_column(table, column)
