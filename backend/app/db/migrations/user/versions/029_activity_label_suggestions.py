"""Add ``activities.label_suggestions`` (issue #63).

Commute detection proposes a label; the athlete confirms it. That needs a place
to hold the proposal which is *not* ``activities.labels``, because writing a
guess there has two visible consequences: the ``commuter`` badge counts labelled
activities, and the RPE queue excludes them — so an early write would both mint
achievement tiers off a heuristic and remove the ride from the prompt where the
athlete would have confirmed it.

Persisted rather than derived on read so a **dismissal is durable**. Re-offering
a suggestion the athlete has already declined, every time the rules are
evaluated, would be worse than not detecting anything.

Nullable and un-backfilled: NULL means "nothing has been suggested for this
activity", which is true of every activity that exists when this lands. The
history scan (``POST /api/activities/scan-commutes``) is how an athlete asks for
their back catalogue to be looked at, rather than a migration deciding for them.

Idempotent: safe to run against DBs that already have the column (including ones
built fresh by SQLAlchemy create_all, which adds it but doesn't stamp alembic).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "029_activity_label_suggestions"
down_revision = "028_achievements_dirty"
branch_labels = None
depends_on = None

_TABLE = "activities"
_COLUMN = "label_suggestions"


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
    return any(row[1] == column_name for row in rows)


def upgrade() -> None:
    conn = op.get_bind()
    if not _column_exists(conn, _TABLE, _COLUMN):
        op.add_column(_TABLE, sa.Column(_COLUMN, sa.JSON(), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    if _column_exists(conn, _TABLE, _COLUMN):
        op.drop_column(_TABLE, _COLUMN)
