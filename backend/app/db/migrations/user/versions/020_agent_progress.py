"""Add the agentic coach's progress columns (issue #43).

The agent loop spends its first few round trips calling tools and producing no
assistant prose, so the frontend's poll — which reads the prose column every
1500 ms and renders whatever partial paragraphs have landed — would show a bare
spinner for the whole tool phase and then dump a finished answer. These columns
carry a *progress code* from a fixed vocabulary (``thinking``,
``tool.get_power_profile``, …) that the web app localises, committed on the same
cadence as the text so a mid-run poll shows real movement.

Separate columns rather than a structured envelope inside ``training_status`` /
``analysis`` on purpose: the frontend parses those as raw prose
(``parseMoodAndParagraphs``), and the activity page, the dashboard card and the
goal-guidance card all share that parser. A new column keeps that contract
untouched. Both are cleared once the prose starts, so a settled row looks
exactly as it did before this migration.

Idempotent: safe to run against DBs that already have the columns (including
ones built fresh by SQLAlchemy create_all, which adds them but doesn't stamp
alembic).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "020_agent_progress"
down_revision = "019_drop_rest_day_activity_links"
branch_labels = None
depends_on = None

_COLUMNS: tuple[tuple[str, str, sa.types.TypeEngine], ...] = (
    ("athletes", "training_status_progress", sa.String()),
    ("activities", "analysis_progress", sa.String()),
)


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
    return any(row[1] == column_name for row in rows)


def upgrade() -> None:
    conn = op.get_bind()
    for table, name, type_ in _COLUMNS:
        if not _column_exists(conn, table, name):
            op.add_column(table, sa.Column(name, type_, nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    for table, name, _type in _COLUMNS:
        if _column_exists(conn, table, name):
            op.drop_column(table, name)
