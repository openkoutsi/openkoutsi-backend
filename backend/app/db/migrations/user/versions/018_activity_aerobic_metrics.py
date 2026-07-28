"""Add aerobic-metric columns to the activities table (issue #37).

Stores the aerobic decoupling figure (or a reason code explaining why one would
be misleading for this ride) and the CP/W' snapshot the activity's ``w_bal``
stream was integrated with. The CP/W' pair is frozen at processing time, in the
same spirit as ``zone_times``, so a ride's W' story doesn't silently change as
the athlete's power curve moves.

Efficiency factor and variability index are deliberately *not* stored: both are
pure ratios of columns already on the row (weighted power / avg HR and weighted
power / avg power), so they are derived on read instead of denormalised here.

Idempotent: safe to run against DBs that already have the columns (including
ones built fresh by SQLAlchemy create_all, which adds them but doesn't stamp
alembic).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "018_activity_aerobic_metrics"
down_revision = "017_message_title_body"
branch_labels = None
depends_on = None

_COLUMNS: tuple[tuple[str, sa.types.TypeEngine], ...] = (
    ("decoupling_pct", sa.Float()),
    ("decoupling_reason", sa.String()),
    ("cp_w", sa.Float()),
    ("w_prime_j", sa.Float()),
)


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
    return any(row[1] == column_name for row in rows)


def upgrade() -> None:
    conn = op.get_bind()
    for name, type_ in _COLUMNS:
        if not _column_exists(conn, "activities", name):
            op.add_column("activities", sa.Column(name, type_, nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    for name, _type in _COLUMNS:
        if _column_exists(conn, "activities", name):
            op.drop_column("activities", name)
