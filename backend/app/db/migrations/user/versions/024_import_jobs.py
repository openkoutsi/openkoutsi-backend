"""Add import_jobs and ActivitySource.format (issue #36).

Bulk import needs somewhere to report progress from, and to record which files
did not make it and why: a Strava export is thousands of files and tens of
minutes of parsing, so the endpoint returns a job id and the client polls it.

``activity_sources.format`` is the other half: originals are stored in the format
they arrived in (``fit``, ``gpx``, ``tcx``) rather than converted, so download and
reprocess know which they are looking at. Existing rows with a file are
backfilled to ``fit``, the only thing they could be.

Idempotent, like every migration in this tree.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "024_import_jobs"
down_revision = "023_sync_leases"
branch_labels = None
depends_on = None


def _table_exists(conn, table_name: str) -> bool:
    row = conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='table' AND name=:n"),
        {"n": table_name},
    ).fetchone()
    return row is not None


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

    if not _table_exists(conn, "import_jobs"):
        op.create_table(
            "import_jobs",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("athlete_id", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False, server_default="pending"),
            sa.Column("source_name", sa.String(), nullable=True),
            sa.Column("total_files", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("imported", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("skipped_duplicate", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("failed", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("results", sa.JSON(), nullable=True),
            sa.Column("error", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["athlete_id"], ["athletes.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _index_exists(conn, "ix_import_jobs_athlete_created"):
        op.create_index(
            "ix_import_jobs_athlete_created",
            "import_jobs",
            ["athlete_id", "created_at"],
        )

    if not _column_exists(conn, "activity_sources", "format"):
        op.add_column("activity_sources", sa.Column("format", sa.String(), nullable=True))

    # Every stored original that predates this column is a FIT — there was no
    # other way to get a file in. Rows with no file stay NULL.
    conn.execute(
        text(
            "UPDATE activity_sources SET format = 'fit' "
            "WHERE format IS NULL AND fit_file_path IS NOT NULL"
        )
    )


def downgrade() -> None:
    conn = op.get_bind()
    if _index_exists(conn, "ix_import_jobs_athlete_created"):
        op.drop_index("ix_import_jobs_athlete_created", table_name="import_jobs")
    if _table_exists(conn, "import_jobs"):
        op.drop_table("import_jobs")
    if _column_exists(conn, "activity_sources", "format"):
        op.drop_column("activity_sources", "format")
