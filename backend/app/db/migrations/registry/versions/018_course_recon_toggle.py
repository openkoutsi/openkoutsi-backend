"""Course recon instance toggle (issue #56).

Adds ``instance_settings.allow_course_recon`` — boolean, non-null, **default
false**. That default is the opposite of ``allow_mcp_server`` (013) and
``allow_personal_access_tokens`` (012), and the difference is the point: those
two publish an interface over data the caller's own credential already reaches,
so there is nothing to decide. This one gates a feature whose distinguishing
half — classifying the road surface under a course — map-matches against OSM
through a Valhalla sidecar the self-hoster runs, builds tiles for, and refreshes
themselves. A full-country tile build was measured at over 5 GB of peak RAM
against a default box with 2 GB, so an instance that has made no decision about
that has not implicitly said yes.

⚠️ **This is visible on upgrade.** Course recon shipped ungated in issue #55, so
an existing deployment using it loses course upload, the segment table and the
pacing plan until an admin turns this on in the admin console under
Settings → "Allow course recon". Nothing is deleted: stored courses, their
segments and their uploaded files stay exactly where they are, remain in the
data export throughout, and come back untouched when the switch is flipped.

Revision ID: 018_course_recon_toggle
Revises: 017_registry_leases
Create Date: 2026-08-31
"""
import sqlalchemy as sa
from alembic import op

revision = "018_course_recon_toggle"
down_revision = "017_registry_leases"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "instance_settings",
        sa.Column(
            "allow_course_recon",
            sa.Boolean(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    with op.batch_alter_table("instance_settings") as batch_op:
        batch_op.drop_column("allow_course_recon")
