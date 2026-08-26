#!/usr/bin/env python3
"""Run Alembic per-user DB migrations for all existing users.

The per-user Alembic env migrates a single database selected by ``USER_ID``.
With one SQLite file per user, upgrading a deployment means running it once per
user; this script automates that loop over ``data/users/*/user.db``.

It runs from the container entrypoint, before uvicorn is exec'd, and that is
deliberate: ``set -e`` plus a non-zero exit here is what stops a container from
serving a database whose schema is behind the code. Checked once, at the only
moment it can change. A runtime guard would be a schema read on every request
for a condition that cannot become false while the process runs.

Which is why what matters is that this be *cheap*, not that it be elsewhere
(issue #50). Two things make it so:

* **One process, not one per user.** This used to spawn a fresh interpreter per
  user through ``subprocess``, so each user cost a Python startup, a full
  ``backend`` package import and an ``env.py`` exec — measured at ~0.89 s each,
  almost none of it migration work. Alembic's Python API re-execs ``env.py`` on
  every ``upgrade`` call, which is all the isolation the per-user env needs.
* **Users already at head are skipped.** On a deploy that ships no user
  migration — most deploys — the loop is one small read per user and nothing
  else.

Not parallelised, on purpose. After the two changes above the steady state is N
cheap reads, and the only deploy that does real work per user is the one
shipping a user migration — precisely the deploy where deterministic ordering
and a readable list of which users failed are worth more than wall-clock time.
If a single migration ever turns out to be slow enough to matter, a ``--jobs``
flag is the escape hatch; a thread pool contending for the same disk is not an
improvement until then.

Usage (from repo root):
    uv run python backend/scripts/migrate_user_dbs.py

Optional flags:
    --dry-run   List user IDs that would be migrated without running alembic
    --data-dir  Override the data directory (default: value from settings / DATA_DIR env var)
"""
import argparse
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional

# Ensure repo root is on sys.path so backend package is importable
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

from backend.app.core.config import settings

ALEMBIC_INI = "backend/alembic-user.ini"


def find_user_ids(data_dir: Path) -> list[str]:
    users_dir = data_dir / "users"
    if not users_dir.is_dir():
        return []
    return sorted(
        d.name
        for d in users_dir.iterdir()
        if d.is_dir() and (d / "user.db").exists()
    )


def build_config() -> Config:
    """One config for the whole run.

    ``env.py`` re-reads ``USER_ID`` from the environment on every exec and sets
    ``sqlalchemy.url`` from it, so a single ``Config`` serves every user.

    ``config_file_name`` is cleared after the first use by :func:`migrate_user`:
    ``env.py`` calls ``fileConfig()`` whenever it is set, which would otherwise
    tear down and rebuild logging once per user.
    """
    return Config(str(REPO_ROOT / ALEMBIC_INI))


def head_revision(cfg: Config) -> Optional[str]:
    """The revision every user database should be at. Read once."""
    return ScriptDirectory.from_config(cfg).get_current_head()


def current_revision(db_path: str) -> Optional[str]:
    """What ``user.db`` says it is at, without going through Alembic.

    ``None`` covers both "no ``alembic_version`` table" and "the table is
    empty". A database created by ``init_user_db`` before stamping was added
    (issue #50) is at head schema-wise but says nothing, and it has to be
    upgraded rather than skipped — every migration in this tree is idempotent
    precisely so that replay is safe.
    """
    try:
        con = sqlite3.connect(db_path)
        try:
            row = con.execute("SELECT version_num FROM alembic_version").fetchone()
        finally:
            con.close()
    except sqlite3.Error:
        return None
    return row[0] if row else None


def migrate_user(cfg: Config, user_id: str, head: Optional[str], dry_run: bool) -> bool:
    """Bring one user's database to head. Returns True on success or a skip."""
    at = current_revision(settings.user_db_path(user_id))
    if head is not None and at == head:
        return True

    print(f"  {'[dry-run] ' if dry_run else ''}upgrading user {user_id} ...", end=" ", flush=True)
    if dry_run:
        print()
        return True

    os.environ["USER_ID"] = user_id
    try:
        command.upgrade(cfg, "head")
    except Exception as exc:
        # In-process now, so this arrives as an exception rather than a return
        # code — but the contract is unchanged: one user's failure costs that
        # user their upgrade, not everyone else's.
        print("FAILED")
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return False
    finally:
        # Only the first exec should configure logging; see `build_config`.
        cfg.config_file_name = None

    print("ok")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate all per-user databases to the latest schema.")
    parser.add_argument("--dry-run", action="store_true", help="List users without running migrations")
    parser.add_argument("--data-dir", default=None, help="Override data directory path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(settings.data_dir)

    user_ids = find_user_ids(data_dir)
    if not user_ids:
        print(f"No user databases found under {data_dir / 'users'}/")
        return

    print(f"Found {len(user_ids)} user(s) in {data_dir / 'users'}/")

    cfg = build_config()
    head = head_revision(cfg)

    previous_user_id = os.environ.get("USER_ID")
    failed = []
    try:
        for user_id in user_ids:
            if not migrate_user(cfg, user_id, head, args.dry_run):
                failed.append(user_id)
    finally:
        # This process may be the entrypoint's, and USER_ID is not ours to keep.
        if previous_user_id is None:
            os.environ.pop("USER_ID", None)
        else:
            os.environ["USER_ID"] = previous_user_id

    print()
    if args.dry_run:
        print("Dry run complete — no changes made.")
    elif failed:
        print(f"Completed with errors. {len(failed)} user(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"All {len(user_ids)} user(s) up to date.")


if __name__ == "__main__":
    main()
