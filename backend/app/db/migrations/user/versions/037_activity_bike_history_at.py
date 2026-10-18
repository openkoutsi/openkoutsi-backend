"""Mark rides attached by "Attach past rides", so the pass can be undone.

``POST /api/bikes/assign-history`` writes ``bike_source = "auto"`` — exactly what
the per-ride automapping on ingest writes — so once it had run there was no
telling a ride it attached from one that arrived already matched, and nothing to
undo it with. ``bike_history_at`` is when the history pass attached the ride;
``DELETE /api/bikes/assign-history`` clears every ride still carrying it.

No backfill: rides attached by a pass before this column existed are
indistinguishable from any other automatic match, and guessing which they were
would undo matches nobody asked to undo.

Idempotent, like every migration in this tree.

Revision ID: 037_activity_bike_history_at
Revises: 036_plan_proposals
Create Date: 2026-10-18
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "037_activity_bike_history_at"
down_revision = "036_plan_proposals"
branch_labels = None
depends_on = None


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
    return any(row[1] == column_name for row in rows)


def upgrade() -> None:
    conn = op.get_bind()
    if not _column_exists(conn, "activities", "bike_history_at"):
        op.add_column(
            "activities",
            sa.Column("bike_history_at", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    conn = op.get_bind()
    if _column_exists(conn, "activities", "bike_history_at"):
        op.drop_column("activities", "bike_history_at")
