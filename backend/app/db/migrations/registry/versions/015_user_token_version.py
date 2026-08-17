"""Session invalidation counter on users (issue #102, F-04).

Adds ``users.token_version`` — integer, non-null, **default 0**. It is stamped
into every access and refresh JWT as ``ver`` and compared on each request, so
raising it ends every session that account has open.

Before this there was no way to invalidate a session at all: the JWTs carry only
``sub``, ``exp`` and ``type``, and the row had nothing to check them against. A
password reset revoked the account's personal access tokens — the handler says
why, in as many words — and left the sessions untouched, so the credentials the
reset was prompted by kept working: the access token for its hour and the
refresh cookie for its full 30 days, minting fresh access tokens the whole time.

Existing users start at 0, which is also what a token minted before this
revision reports (an absent ``ver`` claim reads as 0). That is deliberate: the
upgrade does not sign everyone out, and the first reset after it moves the user
to 1 and takes their old tokens with it. Only this instance's key signs those
tokens, so an absent claim can only mean "issued before the upgrade" — it is not
something a caller can choose.

Revision ID: 015_user_token_version
Revises: 014_provider_connection_refresh_lock
Create Date: 2026-08-16
"""
import sqlalchemy as sa
from alembic import op

revision = "015_user_token_version"
down_revision = "014_provider_connection_refresh_lock"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "token_version",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_column("token_version")
