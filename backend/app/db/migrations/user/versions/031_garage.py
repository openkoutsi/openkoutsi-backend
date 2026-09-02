"""The garage: bikes an athlete owns, rides and maintains (issue #64).

Promotes an existing record rather than adding a parallel one. ``bikes`` was
built by issue #55 as the small equipment concept the pacing physics reads;
this gives it the things a garage needs and links rides to it:

* ``bikes`` — ``odometer_base_km`` (kilometres ridden before openkoutsi saw
  the bike, so wear figures are not systematically low), ``default_sports``
  (which ``sport_type`` values the bike claims, for automapping) and
  ``retired_at`` (a sold bike leaves the pickers but keeps its history —
  deleting would rewrite the athlete's past totals).
* ``activities`` — ``bike_id`` and ``bike_source``. The index on ``bike_id``
  is not optional: every lifetime-distance figure in the garage is a ``SUM``
  filtered on it. ``bike_source`` records *who* chose the bike ("auto" or
  "manual"), which is what stops automapping overwriting a hand correction.
* ``bike_maintenance`` — one row per thing done, keyed by a ``component``
  string so component life is the delta between consecutive entries sharing
  it; ``odometer_km`` is an absolute reading and never moves.
* ``bike_accessories`` — a plain record of what is bolted on. No mass, no
  drag, no coupling to the pacing model (that is deferred, deliberately).

Foreign keys: ``activities.bike_id`` is SET NULL, the two new tables CASCADE
from their bike. As everywhere else in this tree those clauses **document
intent and do not execute** — SQLite only honours them with ``PRAGMA
foreign_keys`` on, which these connections do not set — so ``api/bikes.py``
enforces them explicitly on delete, exactly as it already does for
``Course.bike_id``.

Adds no rows and backfills nothing: every column lands NULL, which reads as
"no bike assigned" and "no baseline", both correct for everything that exists
when this arrives. Assigning history is an explicit athlete request
(``POST /api/bikes/assign-history``), never a migration.

Idempotent, like every migration in this tree: safe against databases already
migrated and against ones built fresh by SQLAlchemy ``create_all``, which
creates the tables but never stamps alembic.

Revision ID: 031_garage
Revises: 030_course_surface
Create Date: 2026-09-01
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy import text

revision = "031_garage"
down_revision = "030_course_surface"
branch_labels = None
depends_on = None

# (table, column, type). Every one nullable — see the module docstring.
_COLUMNS = [
    ("bikes", "odometer_base_km", sa.Float()),
    ("bikes", "default_sports", sa.JSON()),
    ("bikes", "retired_at", sa.DateTime(timezone=True)),
    ("activities", "bike_id", sa.String()),
    ("activities", "bike_source", sa.String()),
]


def _table_exists(conn, table_name: str) -> bool:
    row = conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='table' AND name=:n"),
        {"n": table_name},
    ).fetchone()
    return row is not None


def _index_exists(conn, index_name: str) -> bool:
    row = conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='index' AND name=:n"),
        {"n": index_name},
    ).fetchone()
    return row is not None


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
    return any(row[1] == column_name for row in rows)


def upgrade() -> None:
    conn = op.get_bind()

    for table, column, type_ in _COLUMNS:
        if _table_exists(conn, table) and not _column_exists(conn, table, column):
            op.add_column(table, sa.Column(column, type_, nullable=True))

    # Every per-bike distance in the garage is a SUM over this. Without the
    # index each one is a full scan of the athlete's history.
    if _table_exists(conn, "activities") and not _index_exists(conn, "ix_activities_bike_id"):
        op.create_index("ix_activities_bike_id", "activities", ["bike_id"])

    if not _table_exists(conn, "bike_maintenance"):
        op.create_table(
            "bike_maintenance",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("bike_id", sa.String(), nullable=False),
            sa.Column("athlete_id", sa.String(), nullable=False),
            sa.Column("performed_on", sa.Date(), nullable=False),
            sa.Column("odometer_km", sa.Float(), nullable=True),
            sa.Column("component", sa.String(), nullable=False),
            sa.Column("note", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["bike_id"], ["bikes.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["athlete_id"], ["athletes.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
    if not _index_exists(conn, "ix_bike_maintenance_bike_id"):
        op.create_index("ix_bike_maintenance_bike_id", "bike_maintenance", ["bike_id"])
    if not _index_exists(conn, "ix_bike_maintenance_athlete_id"):
        op.create_index("ix_bike_maintenance_athlete_id", "bike_maintenance", ["athlete_id"])

    if not _table_exists(conn, "bike_accessories"):
        op.create_table(
            "bike_accessories",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("bike_id", sa.String(), nullable=False),
            sa.Column("athlete_id", sa.String(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("note", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["bike_id"], ["bikes.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["athlete_id"], ["athletes.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
    if not _index_exists(conn, "ix_bike_accessories_bike_id"):
        op.create_index("ix_bike_accessories_bike_id", "bike_accessories", ["bike_id"])
    if not _index_exists(conn, "ix_bike_accessories_athlete_id"):
        op.create_index("ix_bike_accessories_athlete_id", "bike_accessories", ["athlete_id"])


def downgrade() -> None:
    conn = op.get_bind()

    for index, table in (
        ("ix_bike_accessories_athlete_id", "bike_accessories"),
        ("ix_bike_accessories_bike_id", "bike_accessories"),
    ):
        if _index_exists(conn, index):
            op.drop_index(index, table_name=table)
    if _table_exists(conn, "bike_accessories"):
        op.drop_table("bike_accessories")

    for index, table in (
        ("ix_bike_maintenance_athlete_id", "bike_maintenance"),
        ("ix_bike_maintenance_bike_id", "bike_maintenance"),
    ):
        if _index_exists(conn, index):
            op.drop_index(index, table_name=table)
    if _table_exists(conn, "bike_maintenance"):
        op.drop_table("bike_maintenance")

    if _index_exists(conn, "ix_activities_bike_id"):
        op.drop_index("ix_activities_bike_id", table_name="activities")

    for table, column, _type in reversed(_COLUMNS):
        if _table_exists(conn, table) and _column_exists(conn, table, column):
            op.drop_column(table, column)
