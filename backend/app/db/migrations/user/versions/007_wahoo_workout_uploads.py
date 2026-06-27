"""Add wahoo_workout_uploads table for tracking structured-workout pushes.

Revision ID: 007_wahoo_workout_uploads
Revises: 006_athlete_training_status_updated_at
Create Date: 2026-06-21

Idempotent: safe to run against DBs that already have the table (e.g. created by
SQLAlchemy create_all after the WahooWorkoutUpload model was added).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "007_wahoo_workout_uploads"
down_revision = "006_athlete_training_status_updated_at"
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

    if not _table_exists(conn, "wahoo_workout_uploads"):
        op.create_table(
            "wahoo_workout_uploads",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("athlete_id", sa.String(), nullable=False),
            sa.Column("workout_definition_id", sa.String(), nullable=True),
            sa.Column("external_id", sa.String(), nullable=False),
            sa.Column("wahoo_plan_id", sa.String(), nullable=True),
            sa.Column("wahoo_workout_id", sa.String(), nullable=True),
            sa.Column("starts", sa.DateTime(timezone=True), nullable=True),
            sa.Column("provider_updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["athlete_id"], ["athletes.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(
                ["workout_definition_id"], ["workout_definitions.id"], ondelete="SET NULL"
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "athlete_id", "external_id", name="uq_wahoo_upload_external"
            ),
        )

    if not _index_exists(conn, "ix_wahoo_workout_uploads_athlete_id"):
        op.create_index(
            "ix_wahoo_workout_uploads_athlete_id",
            "wahoo_workout_uploads",
            ["athlete_id"],
        )


def downgrade() -> None:
    conn = op.get_bind()

    if _index_exists(conn, "ix_wahoo_workout_uploads_athlete_id"):
        op.drop_index(
            "ix_wahoo_workout_uploads_athlete_id", table_name="wahoo_workout_uploads"
        )

    if _table_exists(conn, "wahoo_workout_uploads"):
        op.drop_table("wahoo_workout_uploads")
