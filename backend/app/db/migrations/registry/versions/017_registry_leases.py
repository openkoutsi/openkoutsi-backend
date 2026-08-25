"""Add registry_leases (issue #50).

The three ``lifespan`` pollers — both bridge pollers and the token-expiry sweep
— are ``asyncio`` tasks with no leader election. Two processes therefore run two
of each: the bridge queues get drained twice, and each drain is a real import
and a real LLM bill. That is the assumption `DEPLOY.md` names when it says to
run exactly one process.

``sync_leases`` cannot hold this. It lives in a user's own database, because the
writes it guards land there; who runs the background work for the whole instance
is not any one user's decision, and a lease only means something to holders that
can see the same row. The registry is the database every process opens.

Idempotent, like every migration in this tree: safe to run against DBs already
migrated or created fresh by SQLAlchemy create_all (which builds the tables but
neither stamps alembic).
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
