#!/bin/sh
# Container entrypoint: bring the database schema up to date, then serve, so a
# freshly pulled image is self-applying. Both databases follow the app's
# "create_all on startup + incremental Alembic" model:
#   * Fresh volume    — create_all builds the current schema, so we only STAMP
#     the registry at head to give future migrations a baseline.
#   * Existing volume — the registry carries an alembic_version, so apply
#     pending migrations, then run the per-user migration loop.
# Paths resolve from DATA_DIR, so this operates on the mounted data volume.
set -e

DATA_DIR="${DATA_DIR:-data}"
mkdir -p "$DATA_DIR"

if python - <<'PY'
import sqlite3, sys
from backend.app.core.config import settings
try:
    con = sqlite3.connect(settings.registry_db_path)
    row = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='alembic_version'"
    ).fetchone()
    con.close()
except sqlite3.Error:
    row = None
sys.exit(0 if row else 1)
PY
then
    echo "[entrypoint] Existing registry database — applying migrations ..."
    alembic -c backend/alembic-registry.ini upgrade head

    echo "[entrypoint] Upgrading per-user databases ..."
    python backend/scripts/migrate_user_dbs.py
else
    echo "[entrypoint] Fresh registry database — stamping schema at head ..."
    alembic -c backend/alembic-registry.ini stamp head
fi

echo "[entrypoint] Starting API server ..."
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000
