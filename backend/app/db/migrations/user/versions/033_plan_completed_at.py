"""Record when a training plan finished.

A plan now closes itself once its last scheduled day has passed: ``status``
moves from ``active`` to ``completed``. ``completed_at`` is the timestamp of
that transition, and it is what keeps the closer from fighting the athlete —
reopening a finished plan leaves the timestamp in place, so the next pass sees a
plan that has already been closed once and lets it be. Re-dating a plan clears
it, so a plan whose weeks moved can close again on its new end date.

NULL for every plan that has not finished, which is also what every plan holds
until its end date passes.

Idempotent, like every migration in this tree.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "033_plan_completed_at"
down_revision = "032_decoupling_window"
branch_labels = None
depends_on = None


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
    return any(row[1] == column_name for row in rows)


def upgrade() -> None:
    conn = op.get_bind()
    if not _column_exists(conn, "training_plans", "completed_at"):
        op.add_column(
            "training_plans",
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    conn = op.get_bind()
    if _column_exists(conn, "training_plans", "completed_at"):
        op.drop_column("training_plans", "completed_at")
