"""Add planned_workout_activities join table; drop planned_workouts.completed_activity_id.

Revision ID: 011_planned_workout_activities
Revises: 010_power_best_weight
Create Date: 2026-07-17

Lets several activities jointly complete one planned workout. The scalar
``completed_activity_id`` FK is replaced by a many-to-one join table; any
existing link is copied across before the column is dropped.

Idempotent: safe to run against DBs already migrated or created fresh by
SQLAlchemy create_all (which builds the join table but neither stamps alembic
nor drops the legacy column).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "011_planned_workout_activities"
down_revision = "010_power_best_weight"
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


def upgrade() -> None:
    conn = op.get_bind()

    if not _table_exists(conn, "planned_workout_activities"):
        op.create_table(
            "planned_workout_activities",
            sa.Column("planned_workout_id", sa.String(), nullable=False),
            sa.Column("activity_id", sa.String(), nullable=False),
            sa.ForeignKeyConstraint(
                ["planned_workout_id"], ["planned_workouts.id"], ondelete="CASCADE"
            ),
            sa.ForeignKeyConstraint(
                ["activity_id"], ["activities.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("planned_workout_id", "activity_id"),
            sa.UniqueConstraint(
                "activity_id", name="uq_planned_workout_activities_activity_id"
            ),
        )

    # Copy any existing single-activity links into the join table.
    if _column_exists(conn, "planned_workouts", "completed_activity_id"):
        conn.execute(
            text(
                """
                INSERT OR IGNORE INTO planned_workout_activities
                    (planned_workout_id, activity_id)
                SELECT id, completed_activity_id
                FROM planned_workouts
                WHERE completed_activity_id IS NOT NULL
                """
            )
        )
        # SQLite can't drop a column via plain ALTER — recreate the table.
        with op.batch_alter_table("planned_workouts") as batch_op:
            batch_op.drop_column("completed_activity_id")


def downgrade() -> None:
    conn = op.get_bind()

    if not _column_exists(conn, "planned_workouts", "completed_activity_id"):
        with op.batch_alter_table("planned_workouts") as batch_op:
            batch_op.add_column(
                sa.Column("completed_activity_id", sa.String(), nullable=True)
            )
            batch_op.create_foreign_key(
                "fk_planned_workouts_completed_activity_id",
                "activities",
                ["completed_activity_id"],
                ["id"],
                ondelete="SET NULL",
            )

    if _table_exists(conn, "planned_workout_activities"):
        # Restore the first activity per workout onto the scalar column.
        conn.execute(
            text(
                """
                UPDATE planned_workouts
                SET completed_activity_id = (
                    SELECT activity_id FROM planned_workout_activities
                    WHERE planned_workout_id = planned_workouts.id
                    LIMIT 1
                )
                """
            )
        )
        op.drop_table("planned_workout_activities")
