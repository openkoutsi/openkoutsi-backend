#!/bin/bash

set -e

TARGET=$1
USER=$2

if [ -z "$TARGET" ]; then
    echo "You must pass the server name as the first argument!"
    exit 1
fi

if [ -z "$USER" ]; then
    echo "You must pass the username as the second argument!"
    exit 1
fi

ssh "${USER}@${TARGET}" "sudo -S systemctl stop openkoutsi-backend@${USER}.service"

ssh "${USER}@${TARGET}" "cd projects/openkoutsi-backend && git pull"
ssh "${USER}@${TARGET}" "cd projects/openkoutsi-backend && ~/.local/bin/uv run alembic -c backend/alembic-registry.ini upgrade head"
ssh "${USER}@${TARGET}" "bash -lc 'cd projects/openkoutsi-backend && ~/.local/bin/uv run python backend/scripts/migrate_teams.py'"

ssh "${USER}@${TARGET}" "sudo -S systemctl daemon-reload"
ssh "${USER}@${TARGET}" "sudo -S systemctl start openkoutsi-backend@${USER}.service"
