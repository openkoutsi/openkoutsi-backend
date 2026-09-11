"""Initial third-party API-usage table (issue #66).

One append-only row per outbound call to a third party: every HTTP request to
Strava or Wahoo, and every transactional email. ``endpoint`` holds a normalised
template (``/activities/{id}/streams``), never a raw URL — raw URLs carry
activity ids and, on some paths, query-string credentials.

The four ``ratelimit_*`` columns carry the reading taken from *this* response's
headers, which is what makes "current headroom" a ``ORDER BY created_at DESC
LIMIT 1`` rather than a second table with a second write path. They are null for
services that publish no quota (email, Wahoo). Strava's lower read-only quota
gets its own ``ratelimit_read_*`` pair rather than being merged into the overall
one: a backfill is all reads, so that is the ceiling that binds first.

Revision ID: 001_api_usage
Revises:
Create Date: 2026-09-10
"""
import sqlalchemy as sa
from alembic import op

revision = "001_api_usage"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "api_usage",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("service", sa.String(), nullable=False),
        sa.Column("endpoint", sa.String(), nullable=False),
        sa.Column("method", sa.String(), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column("outcome", sa.String(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("ratelimit_usage_short", sa.Integer(), nullable=True),
        sa.Column("ratelimit_limit_short", sa.Integer(), nullable=True),
        sa.Column("ratelimit_usage_daily", sa.Integer(), nullable=True),
        sa.Column("ratelimit_limit_daily", sa.Integer(), nullable=True),
        sa.Column("ratelimit_read_usage_short", sa.Integer(), nullable=True),
        sa.Column("ratelimit_read_limit_short", sa.Integer(), nullable=True),
        sa.Column("ratelimit_read_usage_daily", sa.Integer(), nullable=True),
        sa.Column("ratelimit_read_limit_daily", sa.Integer(), nullable=True),
    )
    op.create_index("ix_api_usage_created_at", "api_usage", ["created_at"])
    op.create_index(
        "ix_api_usage_service_created", "api_usage", ["service", "created_at"]
    )
    op.create_index("ix_api_usage_user_created", "api_usage", ["user_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_api_usage_user_created", table_name="api_usage")
    op.drop_index("ix_api_usage_service_created", table_name="api_usage")
    op.drop_index("ix_api_usage_created_at", table_name="api_usage")
    op.drop_table("api_usage")
