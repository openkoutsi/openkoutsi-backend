"""Add bikes, courses, course_tracks and course_segments (issue #55).

Course recon stores an uploaded GPX course per the Stage 0 decision (issue #54):
the raw file encrypted on disk under an opaque storage key, the thinned track as
a JSON series in ``course_tracks``, and only coordinate-free derived data
elsewhere — ``courses`` carries the metadata, inputs, chart profile and pacing
outcome, ``course_segments`` the per-segment physics. ``bikes`` is the equipment
concept the physics needs (tyre width → Crr, riding position → CdA), a table
rather than athlete fields because the bike changes per event.

Foreign keys from ``courses`` to ``goals`` and ``bikes`` are SET NULL —
deleting either must never destroy a course. Note SQLite only honours those
clauses when ``PRAGMA foreign_keys`` is on, which these connections do not
set, so the API deletes enforce them explicitly; the clauses document intent.

Idempotent, like every migration in this tree.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "025_courses_bikes"
down_revision = "024_import_jobs"
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

    if not _table_exists(conn, "bikes"):
        op.create_table(
            "bikes",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("athlete_id", sa.String(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("tyre_width_mm", sa.Integer(), nullable=True),
            sa.Column("riding_position", sa.String(), nullable=False, server_default="hoods"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["athlete_id"], ["athletes.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
    if not _index_exists(conn, "ix_bikes_athlete_id"):
        op.create_index("ix_bikes_athlete_id", "bikes", ["athlete_id"])

    if not _table_exists(conn, "courses"):
        op.create_table(
            "courses",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("athlete_id", sa.String(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("goal_id", sa.String(), nullable=True),
            sa.Column("bike_id", sa.String(), nullable=True),
            sa.Column("gpx_file_key", sa.String(), nullable=False),
            sa.Column("gpx_file_encrypted", sa.Boolean(), nullable=False, server_default="1"),
            sa.Column("status", sa.String(), nullable=False, server_default="ready"),
            sa.Column("error", sa.String(), nullable=True),
            sa.Column("target_time_s", sa.Integer(), nullable=True),
            sa.Column("start_time", sa.DateTime(timezone=True), nullable=True),
            sa.Column("ftp_w_used", sa.Float(), nullable=True),
            sa.Column("weight_kg_used", sa.Float(), nullable=True),
            sa.Column("distance_m", sa.Float(), nullable=False),
            sa.Column("elevation_gain_m", sa.Float(), nullable=True),
            sa.Column("elevation_loss_m", sa.Float(), nullable=True),
            sa.Column("min_elevation_m", sa.Float(), nullable=True),
            sa.Column("max_elevation_m", sa.Float(), nullable=True),
            sa.Column("profile", sa.JSON(), nullable=True),
            sa.Column("predicted_time_s", sa.Float(), nullable=True),
            sa.Column("intensity", sa.Float(), nullable=True),
            sa.Column("required_intensity", sa.Float(), nullable=True),
            sa.Column("feasible", sa.Boolean(), nullable=True),
            sa.Column("refusal_reason", sa.String(), nullable=True),
            sa.Column("plan", sa.Text(), nullable=True),
            sa.Column("plan_mood", sa.String(), nullable=True),
            sa.Column("plan_status", sa.String(), nullable=True),
            sa.Column("plan_run_id", sa.String(), nullable=True),
            sa.Column("plan_updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["athlete_id"], ["athletes.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["goal_id"], ["goals.id"], ondelete="SET NULL"),
            sa.ForeignKeyConstraint(["bike_id"], ["bikes.id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id"),
        )
    if not _index_exists(conn, "ix_courses_athlete_id"):
        op.create_index("ix_courses_athlete_id", "courses", ["athlete_id"])

    if not _table_exists(conn, "course_tracks"):
        op.create_table(
            "course_tracks",
            sa.Column("course_id", sa.String(), nullable=False),
            sa.Column("points", sa.JSON(), nullable=False),
            sa.ForeignKeyConstraint(["course_id"], ["courses.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("course_id"),
        )

    if not _table_exists(conn, "course_segments"):
        op.create_table(
            "course_segments",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("course_id", sa.String(), nullable=False),
            sa.Column("segment_index", sa.Integer(), nullable=False),
            sa.Column("start_distance_m", sa.Float(), nullable=False),
            sa.Column("end_distance_m", sa.Float(), nullable=False),
            sa.Column("length_m", sa.Float(), nullable=False),
            sa.Column("avg_gradient", sa.Float(), nullable=False),
            sa.Column("elevation_change_m", sa.Float(), nullable=False),
            sa.Column("segment_type", sa.String(), nullable=False),
            sa.Column("power_w", sa.Float(), nullable=True),
            sa.Column("speed_ms", sa.Float(), nullable=True),
            sa.Column("duration_s", sa.Float(), nullable=True),
            sa.Column("start_offset_s", sa.Float(), nullable=True),
            sa.Column("speed_capped", sa.Boolean(), nullable=False, server_default="0"),
            sa.ForeignKeyConstraint(["course_id"], ["courses.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("course_id", "segment_index"),
        )


def downgrade() -> None:
    conn = op.get_bind()
    if _table_exists(conn, "course_segments"):
        op.drop_table("course_segments")
    if _table_exists(conn, "course_tracks"):
        op.drop_table("course_tracks")
    if _index_exists(conn, "ix_courses_athlete_id"):
        op.drop_index("ix_courses_athlete_id", table_name="courses")
    if _table_exists(conn, "courses"):
        op.drop_table("courses")
    if _index_exists(conn, "ix_bikes_athlete_id"):
        op.drop_index("ix_bikes_athlete_id", table_name="bikes")
    if _table_exists(conn, "bikes"):
        op.drop_table("bikes")
