"""Record which block of a ride the decoupling figure describes.

Aerobic decoupling is now measured over the longest continuous block of a ride
rather than across the whole of it, so a stop long enough to recover from no
longer sits inside the first-half-against-second-half comparison. Where that
block is shorter than the ride, the figure speaks for the block and the number
of seconds it covers has to travel with it — a drift figure over four hours of a
seven-hour ride is not the ride's, and nothing downstream can tell the
difference without being told.

Set only alongside ``decoupling_pct``: NULL wherever there is no figure, which
is also what every activity processed before this holds until it is reprocessed.

Idempotent, like every migration in this tree.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "032_decoupling_window"
down_revision = "031_garage"
branch_labels = None
depends_on = None


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
    return any(row[1] == column_name for row in rows)


def upgrade() -> None:
    conn = op.get_bind()
    if not _column_exists(conn, "activities", "decoupling_window_s"):
        op.add_column(
            "activities", sa.Column("decoupling_window_s", sa.Integer(), nullable=True)
        )


def downgrade() -> None:
    conn = op.get_bind()
    if _column_exists(conn, "activities", "decoupling_window_s"):
        op.drop_column("activities", "decoupling_window_s")
