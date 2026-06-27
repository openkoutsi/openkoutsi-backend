# Deployment Guide

This guide covers the **backend** (API + bridge services). The web frontend is
deployed separately from the [openkoutsi/openkoutsi-web](https://github.com/openkoutsi/openkoutsi-web)
repository — see its README/deploy docs for the Next.js build and systemd unit.

## Prerequisites

- Python 3.12+ with [uv](https://docs.astral.sh/uv/)
- A reverse proxy with TLS (nginx, Caddy, etc.) for production

---

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
WAHOO_BRIDGE_SECRET=               # shared secret — must match BRIDGE_SECRET in wahoo_bridge/.env

# Server-side LLM (OpenAI-compatible) — fallback when no instance/user LLM is configured
LLM_BASE_URL=                      # e.g. http://localhost:11434/v1 or https://api.openai.com/v1
LLM_API_KEY=
LLM_MODEL=                         # e.g. llama3.2, gpt-4o-mini

# Optional: comma-separated list of allowed LLM base URLs users may choose from.
# When set, users must pick from this list. Leave blank to allow any URL.
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

> **Upgrading from a multi-team (v1) deployment?** openkoutsi v2 removes the team
> layer in favour of a single instance with per-user databases. Migrate existing
> team data with the one-time script `backend/scripts/migrate_to_per_user.py`
> (see its module docstring), then run the registry/per-user Alembic migrations.

### First-run setup

On a fresh deployment, navigate to the frontend URL. The setup wizard will appear and guide you through creating the first administrator account. Onboarding is invite-only thereafter: an administrator issues an instance-wide invite from the Admin dashboard, and new users register with that invite token.

### Run

```bash
uv run uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

For production add `--workers 2` (or use gunicorn with uvicorn workers).

---

## 2. Frontend

The web frontend lives in its own repository,
[openkoutsi/openkoutsi-web](https://github.com/openkoutsi/openkoutsi-web). Build
and deploy it from there; point its `NEXT_PUBLIC_API_URL` at the API domain
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
BRIDGE_SECRET=<same random string as WAHOO_BRIDGE_SECRET in main .env>   # python -c "import secrets; print(secrets.token_hex(32))"
WAHOO_WEBHOOK_TOKEN=<token you define in the Wahoo developer portal>      # python -c "import secrets; print(secrets.token_hex(32))"
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

Sending structured workouts to Wahoo (the single-workout "Send to Wahoo" action in the Workouts tab) requires the OAuth scopes `plans_read`, `plans_write`, and `workouts_write`. These are requested automatically; users who connected Wahoo before this feature shipped must reconnect to grant them. The plan-level "Generate workouts" action synthesizes structured workouts server-side via an OpenAI-compatible LLM, so a base URL must be reachable from the backend (resolved athlete → instance → global `LLM_BASE_URL`); it does not upload anything itself — the generated workouts are uploaded to Wahoo individually from the Workouts tab.

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

## 7. Automated Deployment (GitHub Actions)

Pushes to `main` trigger automatic backend deployment:

- **Deploy Backend** — runs when `backend/`, `openkoutsi/`, `pyproject.toml`, or `uv.lock` change

The frontend has its own deploy workflow in the [openkoutsi-web](https://github.com/openkoutsi/openkoutsi-web) repository.

### Required GitHub Secrets

Set these under **Settings → Secrets and variables → Actions** in the repository:

| Secret | Description |
|--------|-------------|
| `VPS_SSH_PRIVATE_KEY` | Private SSH key whose public key is in `~/.ssh/authorized_keys` on the VPS |
| `VPS_HOST` | VPS hostname or IP address |
| `VPS_USER` | Username on the VPS (must match the `@<user>` in the systemd service names) |

### VPS prerequisite: passwordless sudo for systemctl

The deployment scripts run `sudo systemctl` over SSH. The deploy user must be allowed to do so without a password prompt. Create `/etc/sudoers.d/openkoutsi-deploy` on the VPS:

```
<deploy-user> ALL=(ALL) NOPASSWD: /bin/systemctl stop openkoutsi-backend@*.service, /bin/systemctl start openkoutsi-backend@*.service, /bin/systemctl daemon-reload
```

Replace `<deploy-user>` with the actual username. Verify with `sudo visudo -c` after saving.

---

## Checklist

- [ ] `SECRET_KEY` set to a strong random value
- [ ] `ENCRYPTION_KEY` set (required for instance/user LLM API key storage and FIT-file encryption; recommended for all prod deployments)
- [ ] `DATA_DIR` points to a persistent directory (survives restarts/upgrades)
- [ ] `FRONTEND_URL` and `API_URL` point to real domains
- [ ] TLS termination in place for the API (and the frontend, deployed from openkoutsi-web)
- [ ] GitHub Actions secrets set (`VPS_SSH_PRIVATE_KEY`, `VPS_HOST`, `VPS_USER`) if using automated deployment
- [ ] VPS deploy user has passwordless sudo for systemctl (see section 7)
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
