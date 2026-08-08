"""MCP server instance toggle (issue #42).

Adds ``instance_settings.allow_mcp_server`` — boolean, non-null, **default
true**. Like ``allow_personal_access_tokens`` (012) and unlike the LLM and
signup gates, this defaults on: the MCP endpoint publishes read-only, scoped
tools over data the caller's own credential already reaches, so there is no
prior behaviour to preserve and nothing is widened by it being available.

The switch exists so "an AI client may talk to my training data" can be decided
once for the instance rather than per token, and so that the decision lives
somewhere the application can see, test and report — a reverse-proxy rule can
do the same thing but is invisible to the admin console.

Revision ID: 013_mcp_server_toggle
Revises: 012_personal_access_tokens
Create Date: 2026-08-08
"""
import sqlalchemy as sa
from alembic import op

revision = "013_mcp_server_toggle"
down_revision = "012_personal_access_tokens"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "instance_settings",
        sa.Column(
            "allow_mcp_server",
            sa.Boolean(),
            nullable=False,
            server_default="1",
        ),
    )


def downgrade() -> None:
    with op.batch_alter_table("instance_settings") as batch_op:
        batch_op.drop_column("allow_mcp_server")
