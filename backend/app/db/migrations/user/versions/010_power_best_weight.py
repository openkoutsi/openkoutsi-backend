"""Add effective weight / W-per-kg to activity power bests.

Revision ID: 010_power_best_weight
Revises: 009_rename_trademarked_metrics
Create Date: 2026-07-16

Stores the athlete's effective bodyweight at the time of each power best and the
resulting W/kg, so the power curve can be ranked by W/kg (not just watts).  Both
columns are nullable; they are populated during activity processing and by the
weight recompute pass, not by this migration.

Idempotent: each column is only added when it is still missing.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "010_power_best_weight"
down_revision = "009_rename_trademarked_metrics"
branch_labels = None
depends_on = None


_TABLE = "activity_power_bests"
_COLUMNS = ("weight_kg", "w_per_kg")


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
    return any(row[1] == column_name for row in rows)


def upgrade() -> None:
    conn = op.get_bind()
    for column in _COLUMNS:
        if not _column_exists(conn, _TABLE, column):
            op.add_column(_TABLE, sa.Column(column, sa.Float(), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    for column in _COLUMNS:
        if _column_exists(conn, _TABLE, column):
            op.drop_column(_TABLE, column)
