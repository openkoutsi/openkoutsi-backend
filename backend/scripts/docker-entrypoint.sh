#!/bin/sh
# Container entrypoint: bring the database schema up to date, then serve.
#
# This moves the schema upgrades that the old SSH deploy ran on the box
# (scripts/deploy-backend.sh) into container startup, so a freshly pulled image
# is self-applying. Both the registry and per-user databases follow the app's
# "create_all on startup + incremental Alembic" model, so:
#   * Fresh volume  — the app's create_all (registry init / first user.db) builds
#     the current schema on startup; here we only STAMP the registry at head so
#     future migrations track from a known baseline.
#   * Existing volume — the registry already carries an alembic_version, so we
#     apply any pending migrations, then run the per-user migration loop.
# Paths resolve from DATA_DIR (see config / alembic env), so this operates on the
# mounted data volume.
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
