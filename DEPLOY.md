# Deployment Guide

This guide covers the **backend** (API + bridge services). The web frontend is
deployed separately from the [openkoutsi/openkoutsi-web](https://github.com/openkoutsi/openkoutsi-web)
repository — see its README/deploy docs for the Next.js build.

Production runs as **containers** pulled from GHCR (the primary path, below).
The bare-metal/systemd flow is kept as a [documented legacy alternative](#legacy-bare-metal-deployment).

---

## Container deployment (primary)

The deployment model is **build-in-CI, pull-on-VM**:

- CI (`.github/workflows/build-images.yml`) builds the three images and pushes
  them to GHCR. Pull requests build the images to verify they still build, but
  **publishing happens only from `main`** (and manual `workflow_dispatch`).
- The VM only *pulls* images — it never builds, holds source, or accepts an
  inbound CI SSH key. A systemd timer polls GHCR and runs
  `docker compose pull && docker compose up -d`, recreating only the services
  whose image digest changed.

### Images

| Service       | Image                                          | Built from       |
|---------------|------------------------------------------------|------------------|
| Backend       | `ghcr.io/openkoutsi/openkoutsi-backend`        | this repo (`Dockerfile`)            |
| Strava bridge | `ghcr.io/openkoutsi/openkoutsi-strava-bridge`  | this repo (`strava_bridge/`)        |
| Wahoo bridge  | `ghcr.io/openkoutsi/openkoutsi-wahoo-bridge`   | this repo (`wahoo_bridge/`)         |
| Web frontend  | `ghcr.io/openkoutsi/openkoutsi-web`            | [openkoutsi-web](https://github.com/openkoutsi/openkoutsi-web) |

Each build pushes two tags: `latest` (the channel the VM tracks) and
`sha-<shortsha>` (immutable, for rollback — pin a service to a prior `sha-` tag
and `docker compose up -d` to restore it).

### Compose stack, reverse proxy and infrastructure

The `docker-compose.yml`, nginx + certbot, GoAccess dashboard, the `okdeploy`
systemd timer + pull script, and the OpenTofu/cloud-init infrastructure-as-code
(UpCloud, fresh VM, encrypted data volume) all live in the
[openkoutsi/openkoutsi-ops](https://github.com/openkoutsi/openkoutsi-ops)
repository. The whole environment is rebuildable from there. See its README for
the provisioning and cutover runbook.

### GHCR auth on the VM

The images can be made public (no auth to pull) or pulled with a **read-only**
GHCR token (`docker login ghcr.io` with a PAT that has only `read:packages`).
Use a read-only token so a VM compromise cannot push images.

### Secrets (Docker secrets, not `.env`)

All three services read their secret fields from files under `/run/secrets/`
(pydantic-settings `secrets_dir`). Compose mounts one file per secret, named for
the lowercase settings field:

- backend: `secret_key`, `encryption_key`, `strava_client_secret`,
  `bridge_secret`, `wahoo_client_secret`, `wahoo_bridge_secret`
- strava bridge: `strava_client_secret`, `bridge_secret`
- wahoo bridge: `wahoo_bridge_secret`, `wahoo_webhook_token`

Non-secret config (domains, OAuth client *IDs*, `*_BRIDGE_URL`,
`LLM_ALLOWED_SERVERS`) is passed as plain `environment:`. Secret fields are
**never** set as environment variables in containers, so they are not exposed
via `docker inspect` or `/proc/<pid>/environ`. Env vars still take precedence
over file secrets, so set only one source per field.

### Persistent data & encryption

The sensitive SQLite databases (`registry.db`, per-user `user.db` + uploads, and
each bridge `bridge.db`) live on named volumes bound to the VM's **encrypted**
data device. `ENCRYPTION_KEY` (field-level column encryption) is delivered as a
Docker secret, separate from the disk-encryption key — defense in depth.

### Migrations on start

The backend image is **self-applying**: its entrypoint
(`backend/scripts/docker-entrypoint.sh`) runs the registry Alembic upgrade and
the per-user migration loop (`backend/scripts/migrate_user_dbs.py`) against the
mounted data volume before exec'ing uvicorn. No manual migration step is needed
when rolling out a new image.

### Build/run an image locally

```bash
docker build -t openkoutsi-backend .
docker build -t openkoutsi-strava-bridge strava_bridge
docker build -t openkoutsi-wahoo-bridge wahoo_bridge

# Backend needs SECRET_KEY (as a file secret) and a data volume:
mkdir -p /tmp/ok-secrets && python -c "import secrets;print(secrets.token_hex(32))" > /tmp/ok-secrets/secret_key
docker run --rm -p 8000:8000 \
  -v "$PWD/data:/data" -e DATA_DIR=/data \
  -v /tmp/ok-secrets/secret_key:/run/secrets/secret_key:ro \
  openkoutsi-backend
curl localhost:8000/api/health   # {"status":"ok"} once migrations finish
```

---

## Legacy bare-metal deployment

> The sections below describe the original **bare-metal / systemd** deployment.
> The container path above is now primary; this remains for reference and local
> development.

## 1. Backend

### Install dependencies

```bash
uv sync
```

### Configure environment

Create `.env` in the project root:

```env
# Required
SECRET_KEY=<hex-64-chars>          # python -c "import secrets; print(secrets.token_hex(32))"

# Optional – defaults shown
DATA_DIR=data                      # root directory; holds registry.db and users/ (per-user DBs + uploads)
FRONTEND_URL=https://your-domain
API_URL=https://api.your-domain
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=30

# Encryption for stored OAuth tokens, FIT files and instance/user LLM API keys (required for AI features)
ENCRYPTION_KEY=<fernet-key>        # python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Strava (see "Strava Bridge" section)
STRAVA_CLIENT_ID=
STRAVA_CLIENT_SECRET=
BRIDGE_URL=
BRIDGE_SECRET=

# Wahoo (register at developers.wahooligan.com — see "Wahoo Bridge" section)
WAHOO_CLIENT_ID=
WAHOO_CLIENT_SECRET=
WAHOO_BRIDGE_URL=                  # public URL of the Wahoo bridge, e.g. https://wahoo-bridge.your-domain
WAHOO_BRIDGE_SECRET=               # shared secret — must match WAHOO_BRIDGE_SECRET in wahoo_bridge/.env

# Optional: comma-separated allow-list of LLM base URLs users may bring (BYOK).
# When set, BYOK URLs are restricted to this list (at save and use time). Leave
# blank to allow any URL (SSRF guards still apply).
LLM_ALLOWED_SERVERS=               # e.g. http://localhost:11434/v1,https://api.openai.com/v1
```

### Initialize the database

Tables are created automatically on first startup — no manual step required:

```bash
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Migrating existing user databases

New per-user databases are always created with the latest schema. Existing per-user databases require Alembic migrations when upgrading. Run once per user after updating the code:

```bash
USER_ID=<user-uuid> uv run alembic -c backend/alembic-user.ini upgrade head
```

You can find your user UUIDs by listing `data/users/`. To upgrade **all** per-user
databases in one go, use the helper script (it loops over `data/users/*` and runs
the migration for each):

```bash
uv run python backend/scripts/migrate_user_dbs.py        # add --dry-run to preview
```

This step is only needed when upgrading an existing deployment — new installs handle schema creation automatically on first startup.

#### Registry and usage databases

The registry DB and the dedicated **LLM-usage** DB (issue #9) are created
automatically on first startup. When upgrading an existing deployment, apply
their Alembic migrations:

```bash
uv run alembic -c backend/alembic-registry.ini upgrade head   # e.g. adds llm_entitlements
uv run alembic -c backend/alembic-usage.ini upgrade head       # the separate llm_usage DB
```

The usage DB path defaults to `<DATA_DIR>/llm_usage.db`; override it with
`LLM_USAGE_DB`. Its rows are append-only and hold no registry foreign keys, so
it can be pruned/rotated independently.

> **Upgrading from a multi-team (v1) deployment?** openkoutsi v2 removes the team
> layer in favour of a single instance with per-user databases. Migrate existing
> team data with the one-time script `backend/scripts/migrate_to_per_user.py`
> (see its module docstring), then run the registry/per-user Alembic migrations.

### First-run setup

On a fresh deployment, navigate to the frontend URL. The setup wizard will appear and guide you through creating the first administrator account. Onboarding is invite-only thereafter: an administrator issues an instance-wide invite from the Admin dashboard, and new users register with that invite token.

### Run

```bash
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

For production add `--workers 2` (or use gunicorn with uvicorn workers).

---

## 2. Frontend

The web frontend lives in its own repository,
[openkoutsi/openkoutsi-web](https://github.com/openkoutsi/openkoutsi-web). Build
and deploy it from there; point its `API_URL` at the API domain
configured below. Nothing in this repository serves frontend assets.

---

## 3. Reverse Proxy (nginx example)

```nginx
# API
server {
    listen 443 ssl;
    server_name api.your-domain;
    location / { proxy_pass http://127.0.0.1:8000; }
}
```

The frontend has its own `server {}` block — see the openkoutsi-web repository.

---

## 4. Strava Bridge (optional)

The bridge is a separate service that receives Strava webhooks. Strava requires a **public HTTPS URL**.

### Setup

```bash
cd strava_bridge
uv sync
```

Create `strava_bridge/.env`:

```env
STRAVA_CLIENT_SECRET=<same as main app>
BRIDGE_SECRET=<same random string as BRIDGE_SECRET in main .env>   # python -c "import secrets; print(secrets.token_hex(32))"
DATABASE_PATH=bridge.db
```

### Run

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8001
```

Expose it via your reverse proxy (e.g. `bridge.your-domain`) or ngrok for local testing.

### Register webhook with Strava (one-time)

```bash
curl -X POST https://www.strava.com/api/v3/push_subscriptions \
  -F client_id=YOUR_CLIENT_ID \
  -F client_secret=YOUR_CLIENT_SECRET \
  -F callback_url=https://bridge.your-domain/webhook \
  -F verify_token=YOUR_BRIDGE_SECRET
```

A `{"id": N}` response confirms the subscription. Keep the ID to manage the subscription later.

---

## 5. Wahoo Bridge (optional)

The bridge is a separate service that receives Wahoo webhooks. Wahoo requires a **public HTTPS URL**.

### Setup

```bash
cd wahoo_bridge
uv sync
```

Create `wahoo_bridge/.env`:

```env
WAHOO_BRIDGE_SECRET=<same random string as WAHOO_BRIDGE_SECRET in main .env>   # python -c "import secrets; print(secrets.token_hex(32))"
WAHOO_WEBHOOK_TOKEN=<token you define in the Wahoo developer portal>           # python -c "import secrets; print(secrets.token_hex(32))"
DATABASE_PATH=bridge.db
```

### Run

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8085
```

Expose it via your reverse proxy (e.g. `wahoo-bridge.your-domain`).

### Register webhook with Wahoo (one-time)

In the [Wahoo developer portal](https://developers.wahooligan.com), set the webhook URL to:

```
https://wahoo-bridge.your-domain/webhook
```

Set the webhook token to the same value as `WAHOO_WEBHOOK_TOKEN` in `wahoo_bridge/.env`. Wahoo will start sending `workout_summary` events to the bridge immediately.

### Pushing workouts and plans to Wahoo

Sending structured workouts to Wahoo (the single-workout "Send to Wahoo" action in the Workouts tab) requires the OAuth scopes `plans_read`, `plans_write`, and `workouts_write`. These are requested automatically; users who connected Wahoo before this feature shipped must reconnect to grant them. The plan-level "Generate workouts" action synthesizes structured workouts server-side via an OpenAI-compatible LLM, so a base URL must be reachable from the backend (resolved from the athlete's own BYOK config, else the instance's default preset); it does not upload anything itself — the generated workouts are uploaded to Wahoo individually from the Workouts tab.

---

## 6. systemd Services

Service files are provided in the `systemd/` directory as [template units](https://www.freedesktop.org/software/systemd/man/systemd.unit.html#Description). The `@username` suffix at enable time fills in the user and home directory automatically.

```bash
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now openkoutsi-backend@$(whoami)
# Only needed if using the Strava bridge:
sudo systemctl enable --now openkoutsi-bridge@$(whoami)
# Only needed if using the Wahoo bridge:
sudo systemctl enable --now openkoutsi-wahoo-bridge@$(whoami)
```

The units expect the repository to be checked out at `~/projects/openkoutsi-backend`. The frontend systemd unit ships with the [openkoutsi-web](https://github.com/openkoutsi/openkoutsi-web) repository.

Check logs with `journalctl -u openkoutsi-backend@$(whoami) -f` (replace the unit name as needed).

---

## 7. Automated image builds (GitHub Actions)

`.github/workflows/build-images.yml` builds the backend and both bridge images
on every push to `main` and on every pull request:

- **Pull requests:** images are built (to verify they still build) but **not**
  pushed.
- **`main` / `workflow_dispatch`:** images are built and **pushed to GHCR** as
  `latest` + `sha-<shortsha>`.

It logs in to GHCR with the built-in `GITHUB_TOKEN` (`packages: write`), so no
SSH key or VPS secret is stored in the repository — the old SSH `deploy-backend`
workflow has been removed. The VM picks up the new `latest` images via the
polling timer in the [openkoutsi-ops](https://github.com/openkoutsi/openkoutsi-ops)
repository.

The frontend has its own `build-images.yml` in the
[openkoutsi-web](https://github.com/openkoutsi/openkoutsi-web) repository.

---

## Checklist

- [ ] `SECRET_KEY` set to a strong random value
- [ ] `ENCRYPTION_KEY` set (required for instance/user LLM API key storage and FIT-file encryption; recommended for all prod deployments)
- [ ] `DATA_DIR` points to a persistent directory (survives restarts/upgrades)
- [ ] `FRONTEND_URL` and `API_URL` point to real domains
- [ ] TLS termination in place for the API (and the frontend, deployed from openkoutsi-web)
- [ ] Container path: secret files present under `/run/secrets/` (see [Secrets](#secrets-docker-secrets-not-env)); the ops repo provisions these
- [ ] Container path: GHCR pull access configured on the VM (public packages or a read-only token)
- [ ] Completed first-run setup wizard (creates the first admin account)
- [ ] Strava app callback domain updated to production domain (if using Strava)
- [ ] Wahoo webhook URL registered in the developer portal (if using Wahoo)

### Upgrading: zone sync (added in this release)

Zone syncing requires new OAuth scopes. **Existing users who already connected Strava or Wahoo must disconnect and reconnect** to grant the new permissions:

- **Strava** now requests `profile:read_all` (in addition to `read,activity:read_all`) to access athlete zones and FTP.
- **Wahoo** now requests `power_zones_read` (in addition to the existing scopes) to access power zones.

Existing activity syncing is **unaffected** — only zone sync will fail with a "reconnect required" message until the user re-authorises.

### Upgrading: push workouts to Wahoo (added in this release)

Sending structured workouts to Wahoo requires the additional `plans_read`, `plans_write`, and `workouts_write` OAuth scopes. **Existing users who connected Wahoo before this release must disconnect and reconnect** to grant them. Until they do, pushing a workout fails with an `insufficient_scope` error and the UI shows a "reconnect Wahoo" prompt; activity and zone syncing are unaffected.

A per-user table `wahoo_workout_uploads` tracks pushed workouts for idempotent re-pushes. It is created automatically for new per-user databases; existing databases pick it up via the per-user Alembic migration step.
