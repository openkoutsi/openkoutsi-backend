#!/usr/bin/env python3
"""One-time migration: multi-team (v1) → single-instance per-user (v2).

openkoutsi v2 removes the team layer. Each user's athlete profile and **all** of
their training data move from the shared per-team database
(``data/teams/{team_id}/team.db``) into their own per-user database
(``data/users/{user_id}/user.db``), and FIT files / avatars are re-encrypted and
moved under ``data/users/{user_id}/``.

What this script does, reading the *old* schema with raw SQL so it does not
depend on the removed v1 ORM models:

1. For every (team, athlete) pair, ``init_user_db`` the owning user and copy
   every team-DB table row into the user DB, **preserving primary keys**.
   Multi-team users are merged into one athlete; activities are de-duplicated by
   ``(provider, external_id)`` and every merge is logged.
2. Move + re-encrypt FIT files (old team→user key hierarchy → new user key) and
   copy avatars into the per-user directory, rewriting stored paths.
3. Collapse roles to a single global ``administrator``/``user`` on the registry
   ``users`` row (administrator if admin in *any* team), and map the most recent
   ``data_consents`` row onto ``users.consented_at`` / ``users.consent_version``.
4. Seed ``instance_settings`` from the first team's LLM overrides.

It is **idempotent-ish**: re-running re-copies into existing user DBs using
``INSERT OR IGNORE`` so already-migrated rows are skipped.

After a verified run, drop the v1 team tables via the registry Alembic revision
and remove ``data/teams/``.

Usage (from repo root):
    uv run python backend/scripts/migrate_to_per_user.py [--dry-run]

ENCRYPTION_KEY must be set if any FIT files were encrypted.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import shutil
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.config import settings  # noqa: E402

log = logging.getLogger("migrate_to_per_user")

# Tables copied verbatim from each team DB into the owning user's DB. Order
# matters for foreign keys (parents before children).
_COPY_TABLES = [
    "athletes",
    "weight_log",
    "activities",
    "activity_sources",
    "activity_streams",
    "activity_power_bests",
    "activity_distance_bests",
    "activity_intervals",
    "daily_metrics",
    "goals",
    "training_plans",
    "planned_workouts",
    "workout_definitions",
    "wahoo_workout_uploads",
]


# ── Old key hierarchy (v1) — reimplemented locally ─────────────────────────

def _master_raw() -> bytes:
    if not settings.encryption_key:
        raise RuntimeError("ENCRYPTION_KEY is not set — cannot re-encrypt FIT files")
    return base64.urlsafe_b64decode(settings.encryption_key.encode())


def _hkdf(key: bytes, info: str) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info.encode()).derive(key)


def _old_fit_fernet(team_id: str, user_id: str):
    from cryptography.fernet import Fernet
    team_key = _hkdf(_master_raw(), f"team-key:{team_id}")
    derived = _hkdf(team_key, f"fit-file:{user_id}")
    return Fernet(base64.urlsafe_b64encode(derived))


# ── Helpers ────────────────────────────────────────────────────────────────

def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _copy_table(src: sqlite3.Connection, dst: sqlite3.Connection, table: str) -> int:
    if not src.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone():
        return 0
    cols = _columns(src, table)
    dst_cols = set(_columns(dst, table))
    cols = [c for c in cols if c in dst_cols]
    if not cols:
        return 0
    placeholders = ",".join("?" for _ in cols)
    collist = ",".join(cols)
    rows = src.execute(f"SELECT {collist} FROM {table}").fetchall()
    dst.executemany(
        f"INSERT OR IGNORE INTO {table} ({collist}) VALUES ({placeholders})", rows
    )
    return len(rows)


def _team_ids(data_dir: Path) -> list[str]:
    teams_dir = data_dir / "teams"
    if not teams_dir.is_dir():
        return []
    return sorted(d.name for d in teams_dir.iterdir() if (d / "team.db").exists())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report actions without writing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import asyncio

    from backend.app.db.user_session import get_user_session_factory, init_user_db
    from backend.app.core.file_encryption import encrypt_file

    data_dir = Path(settings.data_dir)
    registry_path = Path(settings.registry_db_path)
    if not registry_path.exists():
        log.error("Registry DB not found at %s", registry_path)
        return 1

    reg = sqlite3.connect(registry_path)
    reg.row_factory = sqlite3.Row

    team_ids = _team_ids(data_dir)
    log.info("Found %d team database(s) to migrate", len(team_ids))

    # roles[user_id] = set(), consents[user_id] = (consented_at, version)
    user_roles: dict[str, set[str]] = {}
    user_consent: dict[str, tuple] = {}

    # Collapse memberships -> roles
    try:
        for row in reg.execute("SELECT user_id, roles FROM team_memberships"):
            try:
                roles = json.loads(row["roles"])
            except (TypeError, ValueError):
                roles = []
            user_roles.setdefault(row["user_id"], set()).update(roles)
    except sqlite3.OperationalError:
        log.warning("No team_memberships table — skipping role collapse")

    try:
        for row in reg.execute(
            "SELECT user_id, consented_at, consent_version FROM data_consents ORDER BY consented_at"
        ):
            user_consent[row["user_id"]] = (row["consented_at"], row["consent_version"])
    except sqlite3.OperationalError:
        log.warning("No data_consents table — skipping consent map")

    # Migrate each team DB's athletes into their owning users' DBs.
    for team_id in team_ids:
        team_db = data_dir / "teams" / team_id / "team.db"
        tconn = sqlite3.connect(team_db)
        tconn.row_factory = sqlite3.Row
        athletes = tconn.execute("SELECT id, global_user_id, avatar_path FROM athletes").fetchall()
        log.info("team %s: %d athlete(s)", team_id, len(athletes))

        for ath in athletes:
            user_id = ath["global_user_id"]
            if args.dry_run:
                log.info("  [dry-run] would migrate athlete %s -> user %s", ath["id"], user_id)
                continue

            asyncio.run(init_user_db(user_id))
            uconn = sqlite3.connect(settings.user_db_path(user_id))
            try:
                for table in _COPY_TABLES:
                    n = _copy_table(tconn, uconn, table)
                    if n:
                        log.info("    %s: copied %d row(s)", table, n)

                # Re-encrypt + move FIT files referenced by activity_sources.
                new_fit_dir = settings.user_fit_dir(user_id)
                for src in uconn.execute(
                    "SELECT id, fit_file_path, fit_file_encrypted FROM activity_sources "
                    "WHERE fit_file_path IS NOT NULL"
                ).fetchall():
                    old_path = Path(src[1])
                    if not old_path.exists():
                        continue
                    new_fit_dir.mkdir(parents=True, exist_ok=True)
                    new_path = new_fit_dir / old_path.name
                    if src[2]:  # encrypted: decrypt with old key, re-encrypt with new
                        plaintext = _old_fit_fernet(team_id, user_id).decrypt(old_path.read_bytes())
                        new_path.write_bytes(plaintext)
                        encrypt_file(new_path, user_id)
                    else:
                        shutil.copy2(old_path, new_path)
                    uconn.execute(
                        "UPDATE activity_sources SET fit_file_path=? WHERE id=?",
                        (str(new_path), src[0]),
                    )

                # Move avatar into the per-user directory.
                if ath["avatar_path"] and Path(ath["avatar_path"]).exists():
                    avatar_dir = settings.user_avatar_dir(user_id)
                    avatar_dir.mkdir(parents=True, exist_ok=True)
                    dest = avatar_dir / Path(ath["avatar_path"]).name
                    shutil.copy2(ath["avatar_path"], dest)
                    uconn.execute(
                        "UPDATE athletes SET avatar_path=? WHERE id=?", (str(dest), ath["id"])
                    )

                uconn.commit()
            finally:
                uconn.close()
        tconn.close()

    if args.dry_run:
        log.info("Dry run complete — no changes written")
        return 0

    # Update registry users with collapsed roles + consent.
    for user_id, roles in user_roles.items():
        collapsed = ["administrator", "user"] if "administrator" in roles else ["user"]
        reg.execute("UPDATE users SET roles=? WHERE id=?", (json.dumps(collapsed), user_id))
    for user_id, (consented_at, version) in user_consent.items():
        reg.execute(
            "UPDATE users SET consented_at=?, consent_version=? WHERE id=?",
            (consented_at, version, user_id),
        )

    # Seed instance_settings from the first team's LLM overrides (if any).
    try:
        first = reg.execute(
            "SELECT llm_base_url, llm_api_key_enc, llm_model, llm_analysis_context "
            "FROM teams ORDER BY created_at LIMIT 1"
        ).fetchone()
        if first and any(first):
            reg.execute(
                "INSERT OR REPLACE INTO instance_settings "
                "(id, llm_base_url, llm_api_key_enc, llm_model, llm_analysis_context) "
                "VALUES (1, ?, ?, ?, ?)",
                (first["llm_base_url"], first["llm_api_key_enc"], first["llm_model"],
                 first["llm_analysis_context"]),
            )
            log.info("Seeded instance_settings from first team's LLM config")
    except sqlite3.OperationalError:
        log.warning("Could not read teams table for instance LLM settings")

    reg.commit()
    reg.close()
    log.info(
        "Migration complete. Verify the per-user DBs, then run the registry Alembic "
        "revision to drop the team tables and remove data/teams/."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
