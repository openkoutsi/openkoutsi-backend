"""Rename trademarked training metrics to non-trademarked equivalents.

Revision ID: 009_rename_trademarked_metrics
Revises: 008_goal_guidance
Create Date: 2026-07-16

TrainingPeaks / Peaksware trademarks (NP, TSS, IF, CTL, ATL, TSB) are replaced
with generic names used across the open-source ecosystem:

    activities.normalized_power   -> weighted_power
    activities.tss                -> load
    activities.intensity_factor   -> intensity
    daily_metrics.ctl             -> fitness
    daily_metrics.atl             -> fatigue
    daily_metrics.tsb             -> form
    daily_metrics.tss_day         -> load_day
    planned_workouts.target_tss   -> target_load
    workout_definitions.estimated_tss -> estimated_load

Idempotent: each rename only runs when the old column is still present.
"""
from alembic import op
from sqlalchemy import text

revision = "009_rename_trademarked_metrics"
down_revision = "008_goal_guidance"
branch_labels = None
depends_on = None


_RENAMES = [
    ("activities", "normalized_power", "weighted_power"),
    ("activities", "tss", "load"),
    ("activities", "intensity_factor", "intensity"),
    ("daily_metrics", "ctl", "fitness"),
    ("daily_metrics", "atl", "fatigue"),
    ("daily_metrics", "tsb", "form"),
    ("daily_metrics", "tss_day", "load_day"),
    ("planned_workouts", "target_tss", "target_load"),
    ("workout_definitions", "estimated_tss", "estimated_load"),
]


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
    return any(row[1] == column_name for row in rows)


def _rename(conn, table: str, old: str, new: str) -> None:
    if _column_exists(conn, table, old) and not _column_exists(conn, table, new):
        op.alter_column(table, old, new_column_name=new)


def upgrade() -> None:
    conn = op.get_bind()
    for table, old, new in _RENAMES:
        _rename(conn, table, old, new)


def downgrade() -> None:
    conn = op.get_bind()
    for table, old, new in _RENAMES:
        _rename(conn, table, new, old)
