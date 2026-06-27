"""Add workout_definitions table and workout_definition_id to planned_workouts.

Revision ID: 001_workout_definitions
Revises:
Create Date: 2026-05-06

Idempotent: safe to run against DBs that were created by SQLAlchemy create_all
after the WorkoutDefinition model was added (which creates the table but does not
stamp the alembic version or add the FK column to planned_workouts).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "001_workout_definitions"
down_revision = None
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

    if not _table_exists(conn, "workout_definitions"):
        op.create_table(
            "workout_definitions",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("athlete_id", sa.String(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("sport_type", sa.String(), nullable=False, server_default="Ride"),
            sa.Column("steps", sa.JSON(), nullable=False),
            sa.Column("estimated_duration_s", sa.Integer(), nullable=True),
            sa.Column("estimated_tss", sa.Float(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["athlete_id"], ["athletes.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _index_exists(conn, "ix_workout_definitions_athlete_id"):
        op.create_index(
            "ix_workout_definitions_athlete_id",
            "workout_definitions",
            ["athlete_id"],
        )

    if not _column_exists(conn, "planned_workouts", "workout_definition_id"):
        # SQLite doesn't support ADD CONSTRAINT via ALTER TABLE — use batch mode
        # (copy-and-move) to add the column with its FK in one step.
        with op.batch_alter_table("planned_workouts") as batch_op:
            batch_op.add_column(sa.Column("workout_definition_id", sa.String(), nullable=True))
            batch_op.create_foreign_key(
                "fk_planned_workouts_workout_definition_id",
                "workout_definitions",
                ["workout_definition_id"],
                ["id"],
                ondelete="SET NULL",
            )


def downgrade() -> None:
    conn = op.get_bind()

    if _column_exists(conn, "planned_workouts", "workout_definition_id"):
        with op.batch_alter_table("planned_workouts") as batch_op:
            batch_op.drop_constraint(
                "fk_planned_workouts_workout_definition_id",
                type_="foreignkey",
            )
            batch_op.drop_column("workout_definition_id")

    if _index_exists(conn, "ix_workout_definitions_athlete_id"):
        op.drop_index("ix_workout_definitions_athlete_id", table_name="workout_definitions")

    if _table_exists(conn, "workout_definitions"):
        op.drop_table("workout_definitions")
