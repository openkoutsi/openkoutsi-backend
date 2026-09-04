"""Provider-connection refresh lock (issue #50).

Adds ``provider_connections.refresh_lock_until`` — nullable, no default.

``ensure_fresh_token`` rotates OAuth tokens with a read-modify-write spanning an
``await`` on the provider's refresh endpoint. Nothing serialised it, so two
concurrent syncs for one connection could both read the same refresh token,
present it, and write their result. Wahoo revokes the old refresh token as it
issues the new one, so whichever write lands second stores a dead token and the
connection stays broken until reconnected by hand. Reachable inside a *single*
process; not a multi-replica concern.

The column is the lock: a caller claims the rotation with a conditional UPDATE
(``WHERE refresh_lock_until IS NULL OR refresh_lock_until <= now``) and the
database decides the winner, so the guard holds across processes as well as
tasks. It carries a deadline rather than a boolean so a process dying
mid-rotation releases its claim by expiry instead of wedging the connection.

Revision ID: 014_provider_connection_refresh_lock
Revises: 013_mcp_server_toggle
Create Date: 2026-08-15
"""
import sqlalchemy as sa
from alembic import op

revision = "014_provider_connection_refresh_lock"
down_revision = "013_mcp_server_toggle"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "provider_connections",
        sa.Column("refresh_lock_until", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    with op.batch_alter_table("provider_connections") as batch_op:
        batch_op.drop_column("refresh_lock_until")
