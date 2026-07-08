# openkoutsi-backend

The backend (FastAPI API + bridge services + core library) for openkoutsi, a self-hosted cycling coaching platform. Upload FIT files or sync from Strava/Wahoo, track fitness metrics (CTL/ATL/TSB), and generate periodized training plans from your own server.

> **koutsi** (κουτσί) — Finnish for "coach"

> **Web frontend:** the Next.js UI lives in a separate repository, [openkoutsi/openkoutsi-web](https://github.com/openkoutsi/openkoutsi-web).

## Why

Most cycling coaching tools are cloud-only SaaS. openkoutsi is different: you run it on your own hardware, your data stays under your control, and integrations are optional.

## Features

- **Single instance, per-user data** — one deployment; every user's athlete profile and all training data live in their own isolated SQLite database
- **Invite-only signup** — the setup wizard creates the first administrator; further accounts are created by registering with an instance-wide invite issued by an admin
- **Admin inbox** — in-app messages notify admins about events (e.g. used invites); each user has an isolated per-user message store, deletions are permanent, and the design leaves a hook for future email/push delivery
- **Admin dashboard** — manage users, invitations, password resets, an admin-contact shown on the password-reset page, and instance-wide LLM settings
- **FIT file ingestion** — upload activities directly with automatic TSS, normalized power, and zone distribution analysis
- **Manual activity entry** — log workouts by hand (date, duration, distance, avg/max HR, avg power, cadence, RPE/TSS) with every field optional, behaving like a `manual` data provider
- **Workout categorization** — automatic Coggan-style zone classification with manual override
- **Strava + Wahoo sync** — OAuth integrations with history import and webhook updates through bridge services
- **Zone sync** — sync HR/power zones and FTP from connected providers
- **FTP estimation** — estimate FTP from your power curve via the 20-minute (95%) or Critical Power method, shown on the Power view, and accept either to set your profile FTP
- **Fitness metrics** — CTL/ATL/TSB computed and shown as interactive charts; stale metrics caused by deleted activities are detected and corrected automatically on dashboard load. The fitness history card also shows cycling totals — number of activities, active time, and covered distance — for the selected time period
- **Training calendar** — dashboard calendar shows both performed and planned workouts with distinct visual markers (completed, pending, skipped), and lets you mark a planned workout as done or skipped straight from the day view without opening the plan
- **Training plan generation** — periodized plans (Base → Build → Peak → Taper)
- **Training plan editing** — edit plan metadata (name, goal, start date, length), edit/add/delete individual planned workouts from the calendar day view, and regenerate a plan's workouts (rule-based or AI); completed workouts are locked from edits and preserved on regeneration
- **Activity → plan linking** — uploaded activities are automatically matched to the day's planned workout (sport, TSS ≥ 60%, duration ≥ 60%); manual link/unlink via the plan calendar or the dashboard activity calendar
- **Workout skip tracking** — mark planned workouts as skipped with a reason (illness, injury, fatigue, travel, weather, etc.) for accurate training log and LLM coaching context
- **Structured workouts** — create interval workouts and export as Zwift `.zwo` or FIT workout files for head units (FIT export flattens repeat blocks into individual consecutive steps for reliable display on Wahoo/Garmin devices)
- **Push workouts to Wahoo** — send a structured workout straight to a connected Wahoo account as a plan + scheduled workout, so it appears in Planned Workouts on ELEMNT/RIVAL (schedule within today→+6 days; re-pushing updates instead of duplicating)
- **Generate workouts from a plan** — auto-synthesize structured interval workouts (via LLM) for a training plan's upcoming days in one action; generated workouts are cached on the planned workout (already-generated days are skipped, so no extra LLM calls), rest/out-of-window days are skipped, and a per-day result summary shows what was generated, skipped, or failed. The generated workouts appear in the Workouts tab, where you can review, edit, and upload them to Wahoo individually
- **Goals** — set training/event goals with optional target metrics and dates; when marking a goal achieved, record the final achieved value and a free-text outcome note capturing whether the target was reached
- **Activity labels & notes** — tag activities as "race" or "commute" and add free-text notes (included in AI analysis context); the activity list can filter by label, e.g. show only races (`?labels=race`) or hide commutes (`?exclude_labels=commute`)
- **AI coaching analysis** — per-activity analysis and plan support with OpenAI-compatible backends
- **Koutsi daily feedback** — dashboard card with LLM-generated daily training status covering load trends, recovery state, plan adherence, and goal progress; auto-triggers after uploads/syncs when enabled
- **API v2** — token-scoped (no team slug in any path), no trailing slashes on collection roots, a shared pagination envelope across all collections (activities, goals, plans, workouts, messages, admin lists), analytics consolidated under `/api/metrics`, and `PATCH /api/athlete` for partial updates
- **Privacy-first** — export your data and delete your account at any time
- **Cycling-themed 404 page** — localized "Wrong Turn!" not-found page with cycling flavour

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  FastAPI backend (Python · SQLAlchemy · Alembic)                  │
│  (the Next.js frontend lives in openkoutsi/openkoutsi-web)        │
│                                                                    │
│  data/registry.db                 users, invitations, settings      │
│  data/users/{id}/user.db          per-user athlete + training data   │
│  data/users/{id}/uploads/         encrypted FIT files               │
└────────────────────────────────────────────────────────────────────┘
                 ↕ polls for events
       ┌──────────────────────────────┐     ┌──────────────────────────────┐
       │ Strava Bridge (FastAPI)      │     │ Wahoo Bridge (FastAPI)       │
       │ public webhook endpoint       │     │ public webhook endpoint       │
       └──────────────────────────────┘     └──────────────────────────────┘
```

The bridge services are small external webhook receivers. The main app polls them, so the main app can stay private (for example behind NAT) while only bridges are exposed publicly.

## Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12 · FastAPI · SQLAlchemy 2 (async) · Alembic |
| Database | SQLite (WAL mode) |
| Auth | JWT (`python-jose` · `passlib`) |
| FIT parsing | fitdecode |
| Package manager | uv (Python) |

The web frontend (Next.js 15 · TypeScript · Tailwind · Recharts) lives in [openkoutsi/openkoutsi-web](https://github.com/openkoutsi/openkoutsi-web).

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

### Run locally

```bash
# 1. Clone
git clone https://github.com/openkoutsi/openkoutsi-backend.git
cd openkoutsi-backend

# 2. Create backend env
cat > .env <<'ENV'
SECRET_KEY=<random 256-bit key>
FRONTEND_URL=http://localhost:3000
API_URL=http://localhost:8000
# Optional but recommended if you use encrypted token/file storage features:
# ENCRYPTION_KEY=<fernet-key>
ENV

# 3. Install backend deps and run API
uv sync --group dev
uv run uvicorn backend.main:app --reload --port 8000

# 4. Run the web UI (separate repository)
# Follow the setup in https://github.com/openkoutsi/openkoutsi-web
# (point its API_URL at http://localhost:8000).

# 5. First-run setup
# Open the frontend (default http://localhost:3000) and complete the setup wizard.
```

## Environment variables

Main app (`.env`):

```env
# Required
SECRET_KEY=<random 256-bit key>

# Core settings
DATA_DIR=data
FRONTEND_URL=http://localhost:3000
API_URL=http://localhost:8000

# Optional encryption (required for encrypted key/file flows)
ENCRYPTION_KEY=<fernet-key>

# Strava integration (optional)
STRAVA_CLIENT_ID=
STRAVA_CLIENT_SECRET=
BRIDGE_URL=
BRIDGE_SECRET=

# Wahoo integration (optional)
WAHOO_CLIENT_ID=
WAHOO_CLIENT_SECRET=
WAHOO_BRIDGE_URL=
WAHOO_BRIDGE_SECRET=

# Optional server-side LLM defaults
LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL=
LLM_ALLOWED_SERVERS=
```

The backend intentionally does not send a `temperature` parameter, so each
model applies its own default. This keeps thinking-enabled models (e.g. Claude
with extended thinking, via Anthropic's OpenAI-compatible endpoint) — which
reject any temperature other than `1` — working out of the box. Upstream LLM
errors surface the provider's response body in the logs.

Admins can also configure, per instance (Settings → AI / LLM):

- **Several selectable models** — each with its own extra chat-completion body
  params (e.g. `max_tokens` or a thinking/`reasoning_effort` config). Users pick
  one as their saved default; the selected model's body params are applied to
  their requests. Users may also add their own personal models.
- **Extra request headers** — arbitrary headers added to every outbound LLM
  request (e.g. a provider's zero-data-retention header). Instance headers apply
  to everyone; a user's personal headers override on matching keys.

The connection test (Settings → AI / LLM → *Test connection*) sends a small
"hello world" message using the configured headers and the selected model's body
params and confirms a reply comes back — so it also validates ZDR headers and a
thinking config, not just reachability.

The web frontend has its own configuration (`API_URL`, etc.) — see the [openkoutsi-web](https://github.com/openkoutsi/openkoutsi-web) repository.

## Integrations

- **Strava:** configure Strava app credentials in `.env` and deploy `strava_bridge/` to a public HTTPS URL.
- **Wahoo:** configure Wahoo credentials in `.env` and deploy `wahoo_bridge/` to a public HTTPS URL. Pushing structured workouts to Wahoo requires the `plans_read`, `plans_write`, and `workouts_write` scopes; users connected before this feature must reconnect Wahoo to grant them. The "Generate workouts" plan action needs a server-reachable LLM (resolved athlete → instance → global) to synthesize the structured workouts; uploading the generated workouts to Wahoo is then done individually from the Workouts tab.

### Deployment

Production runs as **containers**: CI builds and pushes the backend and both
bridge images to GHCR (`ghcr.io/openkoutsi/openkoutsi-{backend,strava-bridge,wahoo-bridge}`),
and the VM only *pulls* them — a systemd timer polls GHCR and recreates changed
services (no inbound CI→VM SSH). Secrets are delivered as Docker secret files
under `/run/secrets/`. Schema migrations run automatically on container start.
The compose stack, reverse proxy, and infrastructure-as-code live in the
[openkoutsi/openkoutsi-ops](https://github.com/openkoutsi/openkoutsi-ops)
repository.

Detailed production setup, the container image list, GHCR auth, bridge
registration steps, and the legacy bare-metal/systemd path are in
[DEPLOY.md](DEPLOY.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
