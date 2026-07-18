"""Add plan_adherence_daily snapshot table (issue #26).

Stores one daily snapshot of a training plan's adherence score per active plan,
mirroring the ``daily_metrics`` (Fitness/Fatigue/Form) pattern so the adherence
trend is chartable and survives back-computation.

Idempotent: safe to run against DBs already migrated or created fresh by
SQLAlchemy create_all (which builds the table but neither stamps alembic).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "012_plan_adherence_daily"
down_revision = "011_planned_workout_activities"
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

    if not _table_exists(conn, "plan_adherence_daily"):
        op.create_table(
            "plan_adherence_daily",
            sa.Column("athlete_id", sa.String(), nullable=False),
            sa.Column("plan_id", sa.String(), nullable=False),
            sa.Column("date", sa.Date(), nullable=False),
            sa.Column("score", sa.Float(), nullable=True),
            sa.Column("completed", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("missed", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("skipped", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("pending", sa.Integer(), nullable=False, server_default="0"),
            sa.ForeignKeyConstraint(
                ["athlete_id"], ["athletes.id"], ondelete="CASCADE"
            ),
            sa.ForeignKeyConstraint(
                ["plan_id"], ["training_plans.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("athlete_id", "plan_id", "date"),
        )


def downgrade() -> None:
    conn = op.get_bind()
    if _table_exists(conn, "plan_adherence_daily"):
        op.drop_table("plan_adherence_daily")
