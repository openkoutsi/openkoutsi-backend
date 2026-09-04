"""Add registry_leases (issue #50).

The three ``lifespan`` pollers had no leader election, so two processes ran two
of each and drained the bridge queues twice — a real import and a real LLM bill
each time. That is the assumption `DEPLOY.md` names in requiring one process.

``sync_leases`` cannot hold this: it lives in a user's own database, and who
runs the instance's background work is nobody's user-level decision. The
registry is the database every process opens.

Idempotent, like every migration in this tree.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "017_registry_leases"
down_revision = "016_email_change"
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
    if not _table_exists(conn, "registry_leases"):
        op.create_table(
            "registry_leases",
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("holder", sa.String(), nullable=True),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("name"),
        )


def downgrade() -> None:
    conn = op.get_bind()
    if _table_exists(conn, "registry_leases"):
        op.drop_table("registry_leases")
