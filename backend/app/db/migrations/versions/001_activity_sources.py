"""Add activity_sources table; remove source/external_id/duplicate_of_id/fit columns from activities.

Revision ID: 001_activity_sources
Revises:
Create Date: 2026-04-22
"""
from alembic import op
import sqlalchemy as sa

revision = "001_activity_sources"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Create activity_sources table
    op.create_table(
        "activity_sources",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("activity_id", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("external_id", sa.String(), nullable=True),
        sa.Column("fit_file_path", sa.String(), nullable=True),
        sa.Column("fit_file_encrypted", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["activity_id"], ["activities.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("activity_id", "provider",
                            name="uq_activity_sources_activity_provider"),
    )
    op.create_index(
        "ix_activity_sources_provider_external_id",
        "activity_sources",
        ["provider", "external_id"],
    )

    # 2. Data-migrate: for each existing Activity row, create a matching ActivitySource
    #    using the old columns (source, external_id, fit_file_path, fit_file_encrypted).
    conn = op.get_bind()
    activities = conn.execute(
        sa.text(
            "SELECT id, source, external_id, fit_file_path, fit_file_encrypted "
            "FROM activities "
            "WHERE source IS NOT NULL"
        )
    ).fetchall()

    import uuid
    from datetime import datetime, timezone

    for row in activities:
        act_id, source, ext_id, fit_path, fit_enc = row
        conn.execute(
            sa.text(
                "INSERT INTO activity_sources "
                "(id, activity_id, provider, external_id, fit_file_path, fit_file_encrypted, created_at) "
                "VALUES (:id, :activity_id, :provider, :external_id, :fit_file_path, :fit_file_encrypted, :created_at)"
            ),
            {
                "id": str(uuid.uuid4()),
                "activity_id": act_id,
                "provider": source,
                "external_id": ext_id,
                "fit_file_path": fit_path,
                "fit_file_encrypted": 1 if fit_enc else 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    # 3. Drop the now-redundant columns from activities.
    #    SQLite requires batch mode for column removal.
    with op.batch_alter_table("activities") as batch_op:
        batch_op.drop_column("source")
        batch_op.drop_column("external_id")
        batch_op.drop_column("strava_id")
        batch_op.drop_column("duplicate_of_id")
        batch_op.drop_column("fit_file_path")
        batch_op.drop_column("fit_file_encrypted")


def downgrade() -> None:
    # Re-add columns to activities
    with op.batch_alter_table("activities") as batch_op:
        batch_op.add_column(sa.Column("source", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("external_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("strava_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("duplicate_of_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("fit_file_path", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("fit_file_encrypted", sa.Boolean(), nullable=True))

    # Restore data from activity_sources (best-effort: pick first source per activity)
    conn = op.get_bind()
    sources = conn.execute(
        sa.text(
            "SELECT activity_id, provider, external_id, fit_file_path, fit_file_encrypted "
            "FROM activity_sources"
        )
    ).fetchall()

    for row in sources:
        act_id, provider, ext_id, fit_path, fit_enc = row
        conn.execute(
            sa.text(
                "UPDATE activities SET source=:provider, external_id=:ext_id, "
                "fit_file_path=:fit_path, fit_file_encrypted=:fit_enc "
                "WHERE id=:act_id AND source IS NULL"
            ),
            {
                "provider": provider,
                "ext_id": ext_id,
                "fit_path": fit_path,
                "fit_enc": 1 if fit_enc else 0,
                "act_id": act_id,
            },
        )

    # Drop activity_sources
    op.drop_index("ix_activity_sources_provider_external_id", table_name="activity_sources")
    op.drop_table("activity_sources")
