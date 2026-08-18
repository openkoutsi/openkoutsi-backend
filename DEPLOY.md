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

The backend image carries **numpy**, which the per-second stream math and the
power–duration model fits are built on. It adds roughly 60 MB uncompressed to
the runtime layer — no system packages or build tooling are needed for it, since
`uv sync` installs a manylinux wheel.

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
  `bridge_secret`, `wahoo_client_secret`, `wahoo_bridge_secret`,
  `lettermint_api_key` / `euromail_api_key` (only when outbound email is used;
  set the one matching `EMAIL_PROVIDER`)
- strava bridge: `strava_client_secret`, `bridge_secret`
- wahoo bridge: `wahoo_bridge_secret`, `wahoo_webhook_token`

Email (all optional; omit entirely to leave email disabled): `EMAIL_PROVIDER`
(default `lettermint`; `euromail` is also available) and `EMAIL_FROM` are
non-secret config. The provider's API token is a backend secret
(`lettermint_api_key` or `euromail_api_key`, above). The inbound webhook signing
secret (`lettermint_webhook_secret` / `euromail_webhook_secret`) is consumed by
the optional inbound bridge rather than the backend.

Non-secret config (domains, OAuth client *IDs*, `*_BRIDGE_URL`,
`LLM_ALLOWED_SERVERS`, `EMAIL_PROVIDER`, `EMAIL_FROM`) is passed as plain
`environment:`. Secret fields are
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

# Backend needs SECRET_KEY and ENCRYPTION_KEY (as file secrets) and a data volume:
mkdir -p /tmp/ok-secrets
python -c "import secrets;print(secrets.token_hex(32))" > /tmp/ok-secrets/secret_key
python -c "from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())" > /tmp/ok-secrets/encryption_key
docker run --rm -p 8000:8000 \
  -v "$PWD/data:/data" -e DATA_DIR=/data \
  -v /tmp/ok-secrets/secret_key:/run/secrets/secret_key:ro \
  -v /tmp/ok-secrets/encryption_key:/run/secrets/encryption_key:ro \
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
                                   # bulk imports stage archives under users/{id}/uploads/imports/{job}/
                                   # and delete them when the job ends, so size that volume for the
                                   # expanded archive as well as the activity files it leaves behind
FRONTEND_URL=https://your-domain
API_URL=https://api.your-domain
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=30

# REQUIRED. Encryption for stored OAuth tokens, FIT files and instance/user LLM
# API keys. The backend refuses to start without it, because an empty key used
# to mean "store Strava and Wahoo tokens as plaintext" and said nothing.
ENCRYPTION_KEY=<fernet-key>        # python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Accept plaintext provider tokens instead of setting a key. For development
# and throwaway instances only; logs a warning on every start. Do not set this
# on anything holding real accounts.
ALLOW_PLAINTEXT_SECRETS=false

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

# Email (optional) — all email goes through the swappable email module
# (backend/app/services/email/). Leave unset to keep email disabled; features
# that need it stay unavailable rather than erroring.
EMAIL_PROVIDER=lettermint          # provider selection; "lettermint" or "euromail"
EMAIL_FROM=                        # sender address for outbound transactional mail
LETTERMINT_API_KEY=                # Lettermint API token (outbound sending)
LETTERMINT_WEBHOOK_SECRET=         # verifies inbound Lettermint webhooks (used by the optional inbound bridge)
EUROMAIL_API_KEY=                  # EuroMail API token (outbound sending; EMAIL_PROVIDER=euromail)
EUROMAIL_WEBHOOK_SECRET=           # verifies inbound EuroMail webhooks (used by the optional inbound bridge)

# Optional: comma-separated allow-list of LLM base URLs users may bring (BYOK).
# When set, BYOK URLs are restricted to this list (at save and use time). Leave
# blank to allow any URL (SSRF guards still apply).
LLM_ALLOWED_SERVERS=               # e.g. http://localhost:11434/v1,https://api.openai.com/v1

# Required for a SELF-HOSTED model. The LLM base URL is user-supplied, so URLs
# resolving to loopback (127.x, ::1), RFC 1918 / ULA / CGNAT ranges or 0.0.0.0/8
# are refused by default — otherwise any user can point the backend at whatever
# else runs on this host or its network and read the reply. Set this to true if
# your model runs on localhost (Ollama) or on the LAN. Cloud metadata ranges
# (169.254.x, fe80::/10) stay blocked either way. Requests refused by this
# return 403 naming this variable.
LLM_ALLOW_PRIVATE_NETWORKS=false

# Optional: how many agentic Koutsi runs may be in flight at once in this
# process. An agent loop is 3–5 completions instead of one, so a handful of
# concurrent runs against a local model that serialises requests becomes a queue
# nobody is watching. A run that can't get a slot immediately falls back to the
# single-shot prompt rather than waiting — a worse answer now beats a spinner
# until the 30-minute pending timeout. Default 4; lower it to 1–2 for a single
# local GPU, raise it for a hosted provider.
AGENT_MAX_CONCURRENT_RUNS=4

# Conversational Koutsi (issue #44). Chat is the first LLM surface the *athlete*
# can trigger arbitrarily often, and every turn is a full agent run rather than
# one completion — so it carries its own bounds instead of relying on "one ride,
# one analysis". Unlike a background run it has no single-shot prompt to fall
# back to, so a turn that can't get one of the AGENT_MAX_CONCURRENT_RUNS slots
# waits (visibly, as a "queued" state) rather than being refused; the wait is
# bounded so it can't become an endless spinner.
CHAT_QUEUE_WAIT_SECONDS=45
CHAT_MAX_ROUNDS=4
CHAT_MAX_TURNS_PER_DAY=50
CHAT_MAX_TURNS_PER_CONVERSATION=40
CHAT_MAX_MESSAGE_CHARS=4000
CHAT_HISTORY_CHARS=12000
# Minutes without progress before a chat turn is declared dead. Much shorter
# than the daily card's 30: that runs with nobody watching, this has someone
# waiting on it.
CHAT_STUCK_MINUTES=10

# Privacy policy (GDPR). The consent screen links to this URL. It defaults to
# the canonical koutsi.dev policy; if you self-host you are your own data
# controller and should point this at your own privacy policy.
PRIVACY_POLICY_URL=https://koutsi.dev/privacy
```

### Initialize the database

Tables are created automatically on first startup — no manual step required:

```bash
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Migrating the registry database

The registry (`data/registry.db`) holds accounts, invitations, provider
connections, instance settings and personal access tokens. Bare-metal upgrades
apply its migrations with:

```bash
uv run alembic -c backend/alembic-registry.ini upgrade head
```

Container deployments do this automatically — see *Migrations on start* above.

> **Note:** the latest registry migration `012_personal_access_tokens` adds the
> `personal_access_tokens` table and the `instance_settings.allow_personal_access_tokens`
> column. That column defaults to **1 (on)**, including for existing rows: a PAT
> grants strictly less than the session its owner already holds, and defaulting it
> off would mean the feature silently works nowhere until an admin performs an
> action nobody told them about. If you would rather not have long-lived
> credentials on your box, turn it off explicitly — see
> [ADMIN.md](ADMIN.md), *Personal access tokens*. No new environment variables are
> required.

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

> **Note:** the latest per-user migration `019_drop_rest_day_activity_links`
> **deletes rows** — the only per-user migration so far that does. It removes
> `planned_workout_activities` rows whose planned workout is a rest day, which
> earlier versions created by auto-matching activities onto rest days. Those
> links were invisible in the UI but blocked the activity from being linked to
> the session it actually completed. Deleting them changes no adherence score or
> achievement (rest days are excluded from scoring either way), and nothing else
> references the rows, so no recompute is needed afterwards. It is applied
> automatically by the entrypoint's per-user migration loop (or the helper script
> above); no new environment variables are required.
>
> Every deleted row is copied to `planned_workout_activities_dropped_019` in the
> same user DB first, and the number removed is logged. `downgrade()`
> deliberately restores nothing — putting the links back would put the bug back —
> but the snapshot means a restore stays possible:
>
> ```bash
> sqlite3 data/users/<user-uuid>/user.db \
>   "INSERT INTO planned_workout_activities SELECT * FROM planned_workout_activities_dropped_019"
> ```

> **Note:** per-user migration `021_activity_analysis_updated_at` adds a nullable
> `activities.analysis_updated_at` column — the clock a stuck `pending` analysis
> is aged out against, which that surface previously had no equivalent of. It
> adds no rows, deletes none, backfills nothing (a row stranded before the column
> existed reads as timed out, which is the right answer for it), and needs no new
> environment variables. Applied automatically by the entrypoint's per-user
> migration loop, or by the helper script above.
>
> Related runtime behaviour, needing no configuration: on startup the API settles
> every `pending` LLM run left behind by the previous process — training status,
> goal guidance and activity analysis — before it accepts its first request.
> Nothing that writes a `pending` status survives a restart, so those rows are
> dead by definition, and an activity left in one could never be re-analysed.
> Expect a `Settled N LLM run(s) stranded by the last shutdown` line after a
> redeploy that interrupted a generation. The sweep walks `data/users/*/user.db`
> and inherits the same single-process assumption as the bridge pollers.

### Backing up before a migration

Per-user databases run in **WAL mode**, so `cp user.db` on its own can miss
committed transactions still sitting in the `-wal` file. Use SQLite's own backup
command, which checkpoints for you:

```bash
for db in data/users/*/user.db; do sqlite3 "$db" ".backup '$db.bak'"; done
```

If you copy files instead, copy `user.db`, `user.db-wal` and `user.db-shm`
together. Take the backup **before** starting the new container — the entrypoint
applies pending migrations on start, so there is no checkpoint after that.

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

On a fresh deployment, navigate to the frontend URL. The setup wizard will appear and guide you through creating the first administrator account. Thereafter, an administrator issues an instance-wide invite from the Admin dashboard and new users register with that invite token.

Optionally, admins can enable **self-serve email signup** (Settings tab, or `allow_self_signup` via `PATCH /api/admin/settings`). It requires a configured email provider (see *Email* above): users register with an email address, verify it via an emailed link, and the account activates. Invites keep working regardless. With email configured, users can also reset their own passwords via the "Forgot password?" page. See [ADMIN.md](ADMIN.md) for the full account and password-reset flows.

### Run

```bash
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**Run exactly one process — no `--workers`, no gunicorn.** The backend is
single-process by design, and several things depend on that:

- The **bridge pollers** and the **token-expiry sweep** are `lifespan` asyncio
  tasks with no leader election. A poller fetches every unclaimed event, processes
  it, and only then claims it, so a second process reprocesses the same events —
  duplicate activity imports, duplicate LLM analyses, duplicate spend.
- The **stranded-run sweep** settles every `pending` LLM row at startup on the
  premise that nothing writing one survived the last shutdown (the
  `Settled N LLM run(s) stranded by the last shutdown` behaviour noted under
  *Migrating existing user databases* above). With two processes that premise is
  false: starting the second settles the first one's **live** runs.
- **`AGENT_MAX_CONCURRENT_RUNS`** is an in-process counter, so N processes allow
  N times the concurrency you configured against your LLM.
- **Rate limits** (login, password reset, chat, uploads, bulk import, MCP) are
  held in memory, so N processes give each caller N times the intended
  allowance. For login and password reset that is a weakened brute-force
  defence, not just a looser quota.
- **The one-import-at-a-time check** (`POST /api/activities/import` refuses a
  second job while one is pending or running) is a query against the per-user
  database rather than in-process state, so it does hold between processes. A
  job whose process died cannot clear its own status, so the check is a
  *staleness* question: a row untouched for an hour no longer blocks a new
  import. A healthy job commits its progress every 25 files, so that bound is
  far outside anything a live import does — it is crash recovery, not a
  timeout, and it means a killed import never locks an athlete out permanently.

Two things that used to be on this list no longer are. **Duplicate activity
creation** and **OAuth token rotation** are now guarded in the database rather
than in memory — a lease row and a claimed column respectively — so both hold
between processes. They are listed here because they were fixed for what they do
on *one* box: each was a live race between two concurrent syncs inside a single
process, not a multi-replica hypothetical.

The container image already runs one process — its entrypoint execs a single
uvicorn worker. Give the box more CPU/RAM rather than more processes; see
[SCALING.md](https://github.com/openkoutsi/openkoutsi-ops/blob/main/SCALING.md)
in the ops repository.

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

    # Bulk activity import (POST /api/activities/import) accepts up to 500 MB in
    # one request — a Strava bulk export of a long history. nginx defaults to
    # 1 MB and rejects anything larger with a 413 *before* it reaches the API,
    # so without this the endpoint's own limit never gets a say.
    client_max_body_size 512m;
    # Lower it if you want a smaller ceiling; the API's own 500 MB cap is the
    # upper bound, not a floor.
    # A 900-file import is minutes of parsing, but it happens in the background:
    # the request itself only stages the upload and returns a job id, so the
    # default proxy timeouts are fine.

    location / { proxy_pass http://127.0.0.1:8000; }
}
```

The frontend has its own `server {}` block — see the openkoutsi-web repository.

### Sizing the data volume for imports

An import stages its upload under `users/{id}/uploads/imports/{job_id}/`, expands
the archive beside it, and removes the whole directory when the job ends —
whatever the outcome. A directory left behind by a process that died is swept up
when that athlete next starts an import.

The limits are **per job**, and one job is one athlete: an import may expand to
at most 4 GB across at most 20 000 files, with each file capped at 50 MB
(`backend/app/services/activity_archive.py`). There is no *global* bound, so on
a shared instance the worst case is that ceiling times the number of athletes
importing at once — one import at a time per athlete is enforced, several
athletes at once is not. On a single-user deployment this is not worth thinking
about; on a shared one, size the volume for a few concurrent imports rather than
for one, or lower `MAX_TOTAL_BYTES`.

Running out of disk mid-expansion fails the job with "Ran out of disk space
while unpacking the import" rather than reporting every remaining file as
individually corrupt.

### Exposing (or not exposing) the MCP endpoint

`POST /mcp` is the Model Context Protocol tool server (see the README). It ships
**enabled**, and whether it is available is an instance setting rather than a
proxy rule, so the decision lives somewhere the admin console can show you — the
**Allow the MCP server** switch under Settings, next to the personal-access-token
one. Over the API it is the same field:

```bash
# Turn it off for the whole instance
curl -X PATCH https://api.your-domain/api/admin/settings \
  -H "Authorization: Bearer <admin session token>" \
  -H "Content-Type: application/json" \
  -d '{"allow_mcp_server": false}'
```

Off refuses the endpoint outright — handshake included — with a 404 saying so,
rather than letting a client connect to a server that will decline every useful
call. It is the same shape as `allow_personal_access_tokens`, and for the same
reason: "an AI client may talk to my training data" is a decision a self-hoster
makes once, for the box, not per token.

Denying `/mcp` at the proxy still works and is a reasonable belt-and-braces
measure if you also want it unreachable from outside:

```nginx
location = /mcp { return 404; }
```

Either way, note what this does and does not narrow. The MCP endpoint answers
only to a credential this instance already issued, and the same underlying data
is reachable through the ordinary REST routes with the same token. Turning it off
removes an *interface*, not an exposure — what limits what a credential can see
is its scopes.

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

`STRAVA_CLIENT_SECRET` is **required**, and must be the same value the Strava
app uses — the bridge verifies every webhook's `X-Hub-Signature-256` against it
and rejects the request with `401` when the header is missing or wrong. The
bridge sits on a public HTTPS URL, so this signature is the only thing
separating Strava from anyone else who finds the endpoint.

If the secret is unset the bridge cannot authenticate anything, so it fails
closed: `POST /webhook` answers `403` to every request and a warning is logged
at startup. A bridge that returns `403` to Strava is missing this secret.

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
- [ ] `ENCRYPTION_KEY` set — **required**; the backend refuses to start without it unless `ALLOW_PLAINTEXT_SECRETS=true`, which stores Strava and Wahoo OAuth tokens unencrypted and is not for production
- [ ] `ALLOW_PLAINTEXT_SECRETS` **not** set (check it was not carried over from a dev `.env`)
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
