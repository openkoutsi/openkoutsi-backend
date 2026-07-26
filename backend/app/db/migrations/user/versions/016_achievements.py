"""Add achievement_unlocks table (issue #33).

Stores one row per earned achievement tier. The catalogue itself lives in code
(``openkoutsi.achievements``) rather than in rows, so adding a new achievement
later needs no migration — only unlocks are persisted here.

``achieved_on`` is derived from the athlete's history (the day the criterion was
first met) while ``created_at`` records when we first noticed; the recompute
rewrites this table in place, so a tier can also be revoked when the underlying
data goes away.

Idempotent: safe to run against DBs already migrated or created fresh by
SQLAlchemy create_all (which builds the table but neither stamps alembic).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "016_achievements"
down_revision = "015_plan_week_meta"
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

    if not _table_exists(conn, "achievement_unlocks"):
        op.create_table(
            "achievement_unlocks",
            sa.Column("athlete_id", sa.String(), nullable=False),
            sa.Column("achievement_id", sa.String(), nullable=False),
            sa.Column("tier", sa.Float(), nullable=False),
            sa.Column("achieved_on", sa.Date(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("notified", sa.Boolean(), nullable=False, server_default="0"),
            sa.Column("seen", sa.Boolean(), nullable=False, server_default="0"),
            sa.Column("context", sa.JSON(), nullable=True),
            sa.ForeignKeyConstraint(
                ["athlete_id"], ["athletes.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("athlete_id", "achievement_id", "tier"),
        )


def downgrade() -> None:
    conn = op.get_bind()
    if _table_exists(conn, "achievement_unlocks"):
        op.drop_table("achievement_unlocks")
