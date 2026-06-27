"""
Merge two Activity records into one, keeping the higher-priority source's data.

Usage:
    python scripts/merge_activities.py <team.db path> <keep_id> <drop_id>

The activity identified by <keep_id> is preserved with all its existing metrics
and streams. The ActivitySources from <drop_id> are moved over to <keep_id>
(skipping any provider that already has a source on <keep_id>). <drop_id> and
all its associated data are then deleted.

Example:
    python scripts/merge_activities.py \\
        data/teams/cf6aeab1-.../team.db \\
        294be54a-...   \\
        d9bc6841-...
"""

import sqlite3
import sys


def merge(db_path: str, keep_id: str, drop_id: str) -> None:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")

    keep = con.execute("SELECT * FROM activities WHERE id=?", (keep_id,)).fetchone()
    if keep is None:
        print(f"ERROR: keep activity {keep_id} not found")
        sys.exit(1)

    drop = con.execute("SELECT * FROM activities WHERE id=?", (drop_id,)).fetchone()
    if drop is None:
        print(f"ERROR: drop activity {drop_id} not found")
        sys.exit(1)

    if keep["athlete_id"] != drop["athlete_id"]:
        print("ERROR: activities belong to different athletes")
        sys.exit(1)

    keep_sources = con.execute(
        "SELECT provider, external_id FROM activity_sources WHERE activity_id=?", (keep_id,)
    ).fetchall()
    drop_sources = con.execute(
        "SELECT provider, external_id FROM activity_sources WHERE activity_id=?", (drop_id,)
    ).fetchall()

    keep_providers = {r["provider"] for r in keep_sources}

    print(f"Keep : {keep_id}  {keep['start_time']}  sources={list(keep_providers)}")
    print(f"Drop : {drop_id}  {drop['start_time']}  sources={[r['provider'] for r in drop_sources]}")

    with con:
        for src in drop_sources:
            if src["provider"] in keep_providers:
                print(f"  Skip source '{src['provider']}' — keep already has one")
                con.execute(
                    "DELETE FROM activity_sources WHERE activity_id=? AND provider=?",
                    (drop_id, src["provider"]),
                )
            else:
                print(f"  Moving source '{src['provider']}' ({src['external_id']}) to keep")
                con.execute(
                    "UPDATE activity_sources SET activity_id=? WHERE activity_id=? AND provider=?",
                    (keep_id, drop_id, src["provider"]),
                )

        for table in (
            "activity_streams",
            "activity_power_bests",
            "activity_distance_bests",
            "activity_intervals",
        ):
            n = con.execute(
                f"DELETE FROM {table} WHERE activity_id=?", (drop_id,)
            ).rowcount
            if n:
                print(f"  Deleted {n} row(s) from {table}")

        # Remove any sources still pointing at drop
        con.execute("DELETE FROM activity_sources WHERE activity_id=?", (drop_id,))
        con.execute("DELETE FROM activities WHERE id=?", (drop_id,))

    print("Done — committed.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    _, db_path, keep_id, drop_id = sys.argv
    merge(db_path, keep_id, drop_id)
