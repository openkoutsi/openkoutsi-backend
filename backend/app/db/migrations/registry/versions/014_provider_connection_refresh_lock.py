"""Provider-connection refresh lock (issue #50).

Adds ``provider_connections.refresh_lock_until`` — nullable, no default.

``ensure_fresh_token`` rotates OAuth tokens with a read-modify-write that spans
an ``await`` on the provider's refresh endpoint. Nothing serialised it, so two
concurrent syncs for one connection could both read the same refresh token, both
present it, and both write their result. Wahoo revokes the old refresh token the
moment it issues a new one, so whichever write lands second stores a token that
was already dead — and the connection stays broken until the user reconnects by
hand. This is reachable inside a *single* process today; it is not a
multi-replica concern.

The column is the lock. A caller claims the rotation with a conditional UPDATE
(``WHERE refresh_lock_until IS NULL OR refresh_lock_until <= now``) and the
database decides the winner, which makes the guard hold across processes as well
as across tasks — unlike an in-memory lock. It carries a deadline rather than a
boolean so a process that dies mid-rotation releases its claim by expiry instead
of wedging the connection forever.

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
