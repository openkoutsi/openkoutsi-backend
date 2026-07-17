# openkoutsi-backend

The backend (FastAPI API + bridge services + core library) for openkoutsi, a self-hosted cycling coaching platform. Upload FIT files or sync from Strava/Wahoo, track fitness metrics (Fitness/Fatigue/Form), and generate periodized training plans from your own server.

> **koutsi** (κουτσί) — Finnish for "coach"

> **Web frontend:** the Next.js UI lives in a separate repository, [openkoutsi/openkoutsi-web](https://github.com/openkoutsi/openkoutsi-web).

## Why

Most cycling coaching tools are cloud-only SaaS. openkoutsi is different: you run it on your own hardware, your data stays under your control, and integrations are optional.

## Features

- **Single instance, per-user data** — one deployment; every user's athlete profile and all training data live in their own isolated SQLite database
- **Signup** — the setup wizard creates the first administrator; further accounts come from instance-wide invites issued by an admin, or, when an admin enables the `allow_self_signup` toggle and an email provider is configured, from **self-serve email signup** (register with an email → verify it → account activates)
- **Self-serve password reset** — with email configured, users can request a reset link from the "Forgot password?" page (`POST /api/auth/request-password-reset`); admins can still mint reset links directly. Both are single-use and expire in 1 hour
- **Admin inbox** — in-app messages notify admins about events (e.g. used invites); each user has an isolated per-user message store, deletions are permanent, and the design leaves a hook for future email/push delivery
- **Swappable email module** — a single, provider-agnostic seam (`backend/app/services/email/`) for all email: a generic `EmailProvider` interface (outbound `send`, inbound `verify_inbound_signature`/`parse_inbound`) with `LettermintProvider` and `EuromailProvider` (euromail.dev — EU-based, inbound included on its free tier) implementations, provider selection via `EMAIL_PROVIDER`, and self-rendered inline-styled HTML + plain-text bodies. Optional — with no provider configured, email-dependent features stay unavailable rather than erroring
- **Admin dashboard** — manage users, invitations, password resets, an admin-contact shown on the password-reset page, and instance-wide LLM settings
- **FIT file ingestion** — upload activities directly with automatic Load, weighted power, and zone distribution analysis
- **Manual activity entry** — log workouts by hand (date, duration, distance, avg/max HR, avg power, cadence, RPE/Load) with every field optional, behaving like a `manual` data provider
- **Workout categorization** — automatic Coggan-style zone classification with manual override
- **Strava + Wahoo sync** — OAuth integrations with history import and webhook updates through bridge services
- **Zone sync** — sync HR/power zones and FTP from connected providers
- **FTP estimation** — estimate FTP from your power curve via the 20-minute (95%) or Critical Power method, shown on the Power view, and accept either to set your profile FTP
- **Power curve (watts & W/kg)** — `GET /api/metrics/bests/power?metric=watts|wkg` returns the top-3 efforts per duration; the W/kg view ranks by watts-per-kg using the effective bodyweight at the time of each effort (recorded per power best when activities are processed and re-derived when weight history changes), so genuine W/kg PRs surface instead of just watts PRs divided by current weight
- **Power–duration models** — `GET /api/metrics/power-models` fits several models to your power curve (2- and 3-parameter Critical Power, a CP-anchored exponential, and a power law), returning each model's parameters, a sampled curve for plotting, the fit error, and your estimated potential (Neuromuscular Power/Pmax, Anaerobic Capacity, Maximal Aerobic Power and FTP) so modeled curves can be overlaid on the real power curve
- **Experience level** — self-reported athlete experience level (novice, intermediate, experienced, semi-pro, elite) stored on the profile via `PATCH /api/athlete` and fed into the LLM context for plan/workout generation and training-status, activity and goal analysis, so coaching and progression are tailored to the athlete's level
- **Fitness metrics** — Fitness/Fatigue/Form computed and shown as interactive charts; stale metrics caused by deleted activities are detected and corrected automatically on dashboard load. The fitness history card also shows cycling totals — number of activities, active time, and covered distance — for the selected time period
- **Training calendar** — dashboard calendar shows both performed and planned workouts with distinct visual markers (completed, pending, skipped), and lets you mark a planned workout as done or skipped straight from the day view without opening the plan
- **Training plan generation** — periodized plans (Base → Build → Peak → Taper)
- **Training plan editing** — edit plan metadata (name, goal, start date, length), edit/add/delete individual planned workouts from the calendar day view, and regenerate a plan's workouts (rule-based or AI); completed workouts are locked from edits and preserved on regeneration
- **Plan archiving/unarchiving** — creating a new plan only archives existing active plans whose dates overlap it, so plans covering different periods stay active together; archived plans can be reactivated via `POST /api/plans/{id}/unarchive`, which archives any overlapping active plan in turn
- **Activity → plan linking** — uploaded activities are automatically matched to the day's planned workout (sport, Load ≥ 60%, duration ≥ 60%); manual link/unlink via the plan calendar or the dashboard activity calendar
- **Workout skip tracking** — mark planned workouts as skipped with a reason (illness, injury, fatigue, travel, weather, etc.) for accurate training log and LLM coaching context
- **Structured workouts** — create interval workouts and export as Zwift `.zwo` or FIT workout files for head units (FIT export flattens repeat blocks into individual consecutive steps for reliable display on Wahoo/Garmin devices)
- **Push workouts to Wahoo** — send a structured workout straight to a connected Wahoo account as a plan + scheduled workout, so it appears in Planned Workouts on ELEMNT/RIVAL (schedule within today→+6 days; re-pushing updates instead of duplicating)
- **Generate workouts from a plan** — auto-synthesize structured interval workouts (via LLM) for a training plan's upcoming days in one action; generated workouts are cached on the planned workout (already-generated days are skipped, so no extra LLM calls), rest/out-of-window days are skipped, and a per-day result summary shows what was generated, skipped, or failed. The generated workouts appear in the Workouts tab, where you can review, edit, and upload them to Wahoo individually
- **Goals** — set training/event goals with optional target metrics and dates; when marking a goal achieved, record the final achieved value and a free-text outcome note capturing whether the target was reached
- **AI goal guidance** — on demand, an LLM judges how realistic a goal is for its timeline given the athlete's current fitness and trend (a `realistic`/`ambitious`/`unrealistic` verdict) and gives concrete coaching steps to reach it; streamed in the Koutsi coach voice and persisted per goal
- **Activity labels & notes** — tag activities as "race" or "commute" and add free-text notes (included in AI analysis context); the activity list can filter by label, e.g. show only races (`?labels=race`) or hide commutes (`?exclude_labels=commute`)
- **AI coaching analysis** — per-activity analysis and plan support with OpenAI-compatible backends
- **Koutsi daily feedback** — dashboard card with LLM-generated daily training status covering load trends, recovery state, plan adherence, and goal progress; auto-triggers after uploads/syncs when enabled
- **API v2** — token-scoped (no team slug in any path), no trailing slashes on collection roots, a shared pagination envelope across all collections (activities, goals, plans, workouts, messages, admin lists), analytics consolidated under `/api/metrics`, and `PATCH /api/athlete` for partial updates
- **Privacy-first** — explicit GDPR consent for health-data processing (enforced server-side on the ingestion paths: provider connect and manual upload), a configurable privacy-policy link (`PRIVACY_POLICY_URL`, default `koutsi.dev/privacy`) surfaced on the consent screen, and export/delete your data at any time (the export is a complete per-user dump — profile & LLM settings, activities with notes/labels/analysis, plans, goals, structured workouts, daily fitness metrics, personal records, inbox, weight log, and raw FIT files)
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

# Email (optional) — outbound transactional mail + inbound webhook handling go
# through the swappable email module (backend/app/services/email/). All optional:
# with no provider configured, email-dependent features simply stay unavailable.
EMAIL_PROVIDER=lettermint            # provider selection; "lettermint" or "euromail"
EMAIL_FROM=                          # sender address for outbound mail
LETTERMINT_API_KEY=                  # Lettermint API token for sending
LETTERMINT_WEBHOOK_SECRET=           # secret for verifying inbound Lettermint webhooks
EUROMAIL_API_KEY=                    # EuroMail API token for sending (EMAIL_PROVIDER=euromail)
EUROMAIL_WEBHOOK_SECRET=             # secret for verifying inbound EuroMail webhooks

# Optional: restrict which LLM base URLs users may bring (BYOK). Comma-separated;
# empty = users may bring any URL (subject to SSRF guards).
LLM_ALLOWED_SERVERS=

# Optional: path to the dedicated LLM-usage database (per-call token accounting
# for instance-paid calls). Empty = <DATA_DIR>/llm_usage.db.
LLM_USAGE_DB=
```

There are no server-side LLM env-var defaults: all LLM connections are defined
as presets — instance-wide by an admin, or per-user via BYOK (see below).

The backend intentionally does not send a `temperature` parameter, so each
model applies its own default. This keeps thinking-enabled models (e.g. Claude
with extended thinking, via Anthropic's OpenAI-compatible endpoint) — which
reject any temperature other than `1` — working out of the box. Upstream LLM
errors surface the provider's response body in the logs.

**Guaranteed JSON for structured generation.** Training-plan and structured-workout
generation send the provider a strict JSON-schema `response_format` (derived once
from the backend's own pydantic models), so models that support structured outputs
are constrained to the exact shape the parsers accept — across OpenAI, Anthropic,
Mistral and compatible open-weight servers. It's on by default; a provider that
rejects the parameter is detected and the call is transparently retried without it,
and a preset can pre-empt that with `structured_outputs: false`. The existing
prompt instructions + parse-and-retry remain the final safety net.

Admins configure, per instance (Settings → AI / LLM), a **list of selectable
presets** — each a self-contained connection: display name, stable identifier,
base URL, model id, API key, headers and extra chat-completion body params (e.g.
`max_tokens` or a thinking/`reasoning_effort` config). This lets an admin offer
distinct providers (Anthropic, Mistral, …) as presets. **The first preset in the
list is the instance default.** Users pick a preset — the dropdown shows each
preset's display name, but the selection is stored by its stable identifier, so
renaming a display name never breaks existing selections. A user's selected
preset (or BYOK server) is honoured everywhere an LLM is called on their behalf —
the chat proxy **and** the background analysers (activity analysis, training
status) — falling back to the instance default only when they haven't chosen one.

**Bring your own LLM (BYOK).** Any user can instead point openkoutsi at their
own OpenAI-compatible endpoint under Settings → AI / LLM (base URL + model +
optional API key). Once a user sets their own base URL, **only** their own
config is used — the instance's presets and keys are ignored entirely, so an
instance key can never be sent to a user-chosen server. The API key is
Fernet-encrypted per-user at rest and never returned to the browser. When
`LLM_ALLOWED_SERVERS` is set, BYOK URLs are restricted to that allow-list (at
save time and at use time); the SSRF guard always applies.

The connection tests — *Test connection* (admin, instance presets) and *Test
connection* on the user BYOK card (`POST /api/llm/test-my-connection`) — send a
small "hello world" message using the configured headers and the selected
model's body params and confirm a reply comes back, so they also validate ZDR
headers and a thinking config, not just reachability.

**LLM subscription gating + usage tracking (opt-in).** An admin can flip
`llm_requires_subscription` (Settings → AI / LLM) to require an "LLM access"
entitlement to use the *instance's* LLM credentials. It defaults **off**, so
self-hosted behaviour is unchanged until an admin turns it on. When on, users
without an entitlement can still use every LLM feature via BYOK, or receive a
machine-readable `llm_subscription_required` 403 the frontend turns into an
upsell. Admins grant/revoke entitlements per user in the admin console
(`PUT /api/admin/users/{id}/llm-entitlement`); `GET /api/llm/access` is the
frontend's source of truth for a user's state. Independently, every
**instance-paid** LLM call's token usage (input and output counted separately,
plus the provider and model) is recorded in a **separate** database
(`LLM_USAGE_DB`, default `data/llm_usage.db`) so the hoster can compute average
cost per user over any period via `GET /api/admin/llm-usage/summary`
(day/week/month buckets). BYOK calls are never recorded — the user pays their
own provider.

The web frontend has its own configuration (`API_URL`, etc.) — see the [openkoutsi-web](https://github.com/openkoutsi/openkoutsi-web) repository.

## Integrations

- **Strava:** configure Strava app credentials in `.env` and deploy `strava_bridge/` to a public HTTPS URL.
- **Wahoo:** configure Wahoo credentials in `.env` and deploy `wahoo_bridge/` to a public HTTPS URL. Pushing structured workouts to Wahoo requires the `plans_read`, `plans_write`, and `workouts_write` scopes; users connected before this feature must reconnect Wahoo to grant them. The "Generate workouts" plan action needs a server-reachable LLM (resolved athlete → instance → global) to synthesize the structured workouts; uploading the generated workouts to Wahoo is then done individually from the Workouts tab.
- **Disconnecting a provider:** `DELETE /api/integrations/{provider}/disconnect` optionally deletes the imported activities when `delete_data=true` is passed (accepted as a query parameter *or* in the JSON body). The data is deleted and committed *before* the connection is removed, and a failed deletion returns `500` with the connection left in place — the caller is never told the data is gone unless it actually was.

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

## Evaluating LLM providers/models

[`llm-eval/`](llm-eval/) is a standalone [promptfoo](https://www.promptfoo.dev/)
project for comparing LLM providers/models on prompts that mirror the four places
the platform calls an LLM (plan generation, workout synthesis, activity analysis,
training-status). It imports the real prompt builders so the eval never drifts
from production, grades the JSON families objectively and the prose ones via a
side-by-side web UI, and is not wired into the app or CI. See
[llm-eval/README.md](llm-eval/README.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
