#!/usr/bin/env python3
"""Run Alembic per-user DB migrations for all existing users.

The per-user Alembic env migrates a single database selected by ``USER_ID``.
With one SQLite file per user, upgrading a deployment means running it once per
user; this script automates that loop over ``data/users/*/user.db``.

Usage (from repo root):
    uv run python backend/scripts/migrate_user_dbs.py

Optional flags:
    --dry-run   List user IDs that would be migrated without running alembic
    --data-dir  Override the data directory (default: value from settings / DATA_DIR env var)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# Ensure repo root is on sys.path so backend package is importable
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.config import settings


def find_user_ids(data_dir: Path) -> list[str]:
    users_dir = data_dir / "users"
    if not users_dir.is_dir():
        return []
    return sorted(
        d.name
        for d in users_dir.iterdir()
        if d.is_dir() and (d / "user.db").exists()
    )


def migrate_user(user_id: str, dry_run: bool) -> bool:
    """Run `alembic upgrade head` for a single user. Returns True on success."""
    print(f"  {'[dry-run] ' if dry_run else ''}upgrading user {user_id} ...", end=" ", flush=True)
    if dry_run:
        print()
        return True

    env = {**os.environ, "USER_ID": user_id}
    # Invoke alembic through the current interpreter (`python -m alembic`) rather
    # than `uv run` so this works both locally (under `uv run`) and inside the
    # container image, where uv is not installed but alembic is on the venv.
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", "backend/alembic-user.ini", "upgrade", "head"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("ok")
        return True
    else:
        print("FAILED")
        sys.stderr.write(result.stderr)
        return False


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

    failed = []
    for user_id in user_ids:
        if not migrate_user(user_id, args.dry_run):
            failed.append(user_id)

    print()
    if args.dry_run:
        print("Dry run complete — no changes made.")
    elif failed:
        print(f"Completed with errors. {len(failed)} user(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"All {len(user_ids)} user(s) migrated successfully.")


if __name__ == "__main__":
    main()
