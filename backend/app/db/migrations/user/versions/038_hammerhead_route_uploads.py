"""Add hammerhead_route_uploads, linking a course to the Karoo route it was pushed as.

Issue #69. Route sync with Hammerhead is push-only, so the link lives in a join
table rather than on ``courses``: a re-push replaces the route in place, and
deleting the course deletes the route we created.

Idempotent, like every migration in this tree.

Revision ID: 038_hammerhead_route_uploads
Revises: 037_activity_bike_history_at
Create Date: 2026-10-25
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "038_hammerhead_route_uploads"
down_revision = "037_activity_bike_history_at"
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
    if not _table_exists(conn, "hammerhead_route_uploads"):
        op.create_table(
            "hammerhead_route_uploads",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column(
                "athlete_id",
                sa.String(),
                sa.ForeignKey("athletes.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "course_id",
                sa.String(),
                sa.ForeignKey("courses.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("hammerhead_route_id", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.UniqueConstraint(
                "athlete_id", "course_id", name="uq_hammerhead_route_course"
            ),
        )
    if not _index_exists(conn, "ix_hammerhead_route_uploads_athlete_id"):
        op.create_index(
            "ix_hammerhead_route_uploads_athlete_id",
            "hammerhead_route_uploads",
            ["athlete_id"],
        )


def downgrade() -> None:
    conn = op.get_bind()
    if _index_exists(conn, "ix_hammerhead_route_uploads_athlete_id"):
        op.drop_index(
            "ix_hammerhead_route_uploads_athlete_id",
            table_name="hammerhead_route_uploads",
        )
    if _table_exists(conn, "hammerhead_route_uploads"):
        op.drop_table("hammerhead_route_uploads")
