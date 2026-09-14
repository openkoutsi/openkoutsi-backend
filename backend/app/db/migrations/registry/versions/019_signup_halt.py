"""Temporary signup halt, with a reason.

Adds two columns to ``instance_settings``:

* ``signups_halted`` — boolean, non-null, **default false**.
* ``signup_halt_reason`` — nullable text, the admin's own sentence.

Deliberately a second switch rather than a reuse of ``allow_self_signup`` (011).
That one is standing policy — whether this instance offers self-serve signup at
all — and it carries no reason, so a temporary pause expressed through it tells
would-be users "not enabled on this instance", which is both false and unhelpful.
Keeping the two apart also means lifting the halt restores whatever the instance
was doing before, rather than asking the admin to remember it.

Unlike ``allow_personal_access_tokens`` (012), ``allow_mcp_server`` (013) and
``allow_course_recon`` (018), this one stops at the front door rather than
refusing a capability: invitations keep redeeming and an emailed verification
link still activates its account. What it protects is the rate at which
strangers arrive, and an invitation is the admin's own deliberate act.

Nothing is visible on upgrade: the default is off, so an existing deployment
carries on exactly as before until an admin flips it.

Revision ID: 019_signup_halt
Revises: 018_course_recon_toggle
Create Date: 2026-09-14
"""
import sqlalchemy as sa
from alembic import op

revision = "019_signup_halt"
down_revision = "018_course_recon_toggle"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "instance_settings",
        sa.Column(
            "signups_halted",
            sa.Boolean(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "instance_settings",
        sa.Column("signup_halt_reason", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    with op.batch_alter_table("instance_settings") as batch_op:
        batch_op.drop_column("signup_halt_reason")
        batch_op.drop_column("signups_halted")
