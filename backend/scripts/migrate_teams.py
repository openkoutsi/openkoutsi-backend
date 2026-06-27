#!/usr/bin/env python3
"""Run Alembic team DB migrations for all existing teams.

Usage (from repo root):
    uv run python backend/scripts/migrate_teams.py

Optional flags:
    --dry-run   List team IDs that would be migrated without running alembic
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


def find_team_ids(data_dir: Path) -> list[str]:
    teams_dir = data_dir / "teams"
    if not teams_dir.is_dir():
        return []
    return sorted(
        d.name
        for d in teams_dir.iterdir()
        if d.is_dir() and (d / "team.db").exists()
    )


def migrate_team(team_id: str, dry_run: bool) -> bool:
    """Run `alembic upgrade head` for a single team. Returns True on success."""
    print(f"  {'[dry-run] ' if dry_run else ''}upgrading team {team_id} ...", end=" ", flush=True)
    if dry_run:
        print()
        return True

    env = {**os.environ, "TEAM_ID": team_id}
    result = subprocess.run(
        ["uv", "run", "alembic", "-c", "backend/alembic-team.ini", "upgrade", "head"],
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
    parser = argparse.ArgumentParser(description="Migrate all team databases to the latest schema.")
    parser.add_argument("--dry-run", action="store_true", help="List teams without running migrations")
    parser.add_argument("--data-dir", default=None, help="Override data directory path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(settings.data_dir)

    team_ids = find_team_ids(data_dir)
    if not team_ids:
        print(f"No team databases found under {data_dir / 'teams'}/")
        return

    print(f"Found {len(team_ids)} team(s) in {data_dir / 'teams'}/")

    failed = []
    for team_id in team_ids:
        success = migrate_team(team_id, args.dry_run)
        if not success:
            failed.append(team_id)

    print()
    if args.dry_run:
        print("Dry run complete — no changes made.")
    elif failed:
        print(f"Completed with errors. {len(failed)} team(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"All {len(team_ids)} team(s) migrated successfully.")


if __name__ == "__main__":
    main()
