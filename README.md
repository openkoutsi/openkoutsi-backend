# openkoutsi-backend

[![codecov](https://codecov.io/gh/openkoutsi/openkoutsi-backend/graph/badge.svg)](https://codecov.io/gh/openkoutsi/openkoutsi-backend)

The backend (FastAPI API + bridge services + core library) for openkoutsi, a self-hosted cycling coaching platform. Upload FIT files or sync from Strava/Wahoo, track fitness metrics (Fitness/Fatigue/Form), and generate periodized training plans from your own server.

> **koutsi** (κουτσί) — Finnish for "coach"

> **Web frontend:** the Next.js UI lives in a separate repository, [openkoutsi/openkoutsi-web](https://github.com/openkoutsi/openkoutsi-web).

Most cycling coaching tools are cloud-only SaaS. openkoutsi runs on your own hardware, your data stays under your control, and integrations are optional.

**Docs:** [DEPLOY.md](DEPLOY.md) (installation, configuration, upgrades) · [ADMIN.md](ADMIN.md) (running an instance) · [openkoutsi-docs](https://github.com/openkoutsi/openkoutsi-docs) (user guide)

## Features

### Accounts and privacy

- **Per-user data** — one deployment; each user's data lives in its own isolated SQLite database
- **Accounts** — setup wizard for the first admin, then invitations or optional self-serve email signup; email change requires approval from both addresses; self-serve and admin password reset
- **Personal access tokens** — scoped, expiring, revocable tokens for your own scripts and MCP clients
- **Privacy-first** — GDPR consent for health data, full data export and account deletion; stored secrets and activity files are encrypted (`ENCRYPTION_KEY`)
- **AI transparency** — model-written text is flagged in every response so the UI can label it (EU AI Act Art. 50)
- **Inbox** — in-app notifications for achievements, token expiry and admin events

### Activities

- **Ingestion** — FIT upload, bulk import of FIT/GPX/TCX files or a whole Strava export archive, and manual entry
- **No location data** — ride coordinates are never stored; only deliberately uploaded courses keep a route
- **Streams** — 1 Hz streams on a common clock, with derived torque and W′ balance
- **Labels, notes and RPE** — tag rides, add notes, rate perceived effort; commute detection suggests the `commute` label from your own rules

### Metrics and analysis

- **Fitness/Fatigue/Form** — history charts and a forecast from active plans
- **Zones** — weekly time in zone and intensity distribution (polarized, pyramidal, …)
- **Power** — power curve (watts and W/kg), power–duration models and FTP estimation
- **Aerobic metrics** — efficiency factor, variability index and aerobic decoupling
- **Achievements & streaks** — badges and weekly streaks, optional

### Planning

- **Training plans** — periodized plans (Base → Build → Peak → Taper), editable, auto-closed when finished, archivable
- **Calendar and adherence** — planned vs performed workouts, automatic activity linking, skip tracking and adherence scores
- **Structured workouts** — interval workouts exported as `.zwo` or FIT, pushed to Wahoo, or generated from a plan by the LLM
- **Goals** — targets with optional AI guidance on how realistic they are

### Courses and bikes

- **Course recon** — a GPX course becomes a segment table with power targets and predicted splits, solvable for a target time or power, plus a written pacing plan; optional road-surface matching via a self-hosted Valhalla sidecar
- **Garage** — your bikes, their distance, maintenance log and components; rides are assigned to bikes automatically

### AI coaching

- **Activity analysis and daily feedback** — LLM coaching on each ride and a daily training status card
- **Agentic Koutsi** — opt-in agent mode where the coach queries your data through MCP tools
- **Chat** — ask Koutsi questions; it can draft plans or plan changes that you approve
- **Any OpenAI-compatible model** — instance presets set by the admin, or bring your own (BYOK)

### Integrations

- **Strava and Wahoo** — OAuth connection, history import, webhook updates and zone sync
- **Email** — optional, provider-agnostic (Lettermint or EuroMail)

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  FastAPI backend (Python · SQLAlchemy · Alembic)                  │
│  (the Next.js frontend lives in openkoutsi/openkoutsi-web)        │
│                                                                    │
│  data/registry.db                 users, invitations, settings      │
│  data/users/{id}/user.db          per-user athlete + training data   │
│  data/users/{id}/uploads/         encrypted activity files          │
└────────────────────────────────────────────────────────────────────┘
                 ↕ polls for events
       ┌──────────────────────────────┐     ┌──────────────────────────────┐
       │ Strava Bridge (FastAPI)      │     │ Wahoo Bridge (FastAPI)       │
       │ public webhook endpoint       │     │ public webhook endpoint       │
       └──────────────────────────────┘     └──────────────────────────────┘
```

The bridges are small public webhook receivers. The main app polls them, so it can stay private (e.g. behind NAT).

Where to find cross-cutting code:

| Concern | Home |
|---|---|
| Auth, per-user session, athlete lookup | `core/deps.py` |
| LLM calls (streaming / non-streaming) | `services/llm_streaming.py`, `services/llm_client.py` |
| Provider sync and HTTP accounting | `services/provider_sync.py`, `services/providers/http.py`, `services/api_usage.py`, `services/quota.py` |
| Activity file parsing | `openkoutsi/activity_formats.py` + `gpx.py` / `tcx.py` / `fit.py` |
| Bulk import | `services/activity_import.py`, `services/activity_archive.py` |
| Bridge event claiming | `services/bridge_client.py` |
| Cross-process locks and leader election | `db/leases.py`, `services/leadership.py` |
| Creating a user database | `db/user_session.py` (`init_user_db`) |
| MCP tools | `mcp/registry.py`, `mcp/dispatch.py` |

## Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12 · FastAPI · SQLAlchemy 2 (async) · Alembic |
| Database | SQLite (WAL mode) |
| Auth | JWT (`python-jose` · `bcrypt`) |
| FIT parsing | fitdecode |
| Stream & fit math | numpy |
| Package manager | uv |

## Getting Started

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/openkoutsi/openkoutsi-backend.git
cd openkoutsi-backend

cat > .env <<'ENV'
SECRET_KEY=<random 256-bit key>
ENCRYPTION_KEY=<fernet-key>
FRONTEND_URL=http://localhost:3000
API_URL=http://localhost:8000
ENV

uv sync --group dev
uv run uvicorn backend.main:app --reload --port 8000
```

Then run the [web UI](https://github.com/openkoutsi/openkoutsi-web) with its `API_URL` pointing at `http://localhost:8000` and complete the setup wizard.

All environment variables (Strava, Wahoo, email, LLM, Valhalla, …) are documented in [DEPLOY.md](DEPLOY.md). LLM providers are configured in the app, not via environment — see [ADMIN.md](ADMIN.md#llm-configuration).

### Tests

```bash
uv run python -m pytest tests/
```

CI uploads coverage to Codecov, measured with `COVERAGE_CORE=sysmon` (the default tracer under-reports async route handlers).

## Deployment

Production runs as containers: CI publishes the backend and bridge images to GHCR, and the VM pulls them. Migrations run on container start. The compose stack and infrastructure live in [openkoutsi-ops](https://github.com/openkoutsi/openkoutsi-ops); setup details, bridge registration and the legacy bare-metal path are in [DEPLOY.md](DEPLOY.md).

## MCP server

`POST /mcp` is a read-only [Model Context Protocol](https://modelcontextprotocol.io/) server exposing ten coaching tools over your own data (training status, activities, plans, goals, power profile, zones, athlete profile). Tools return computed aggregates, never raw streams or coordinates. Authenticate with a personal access token; its read scopes decide which tools answer.

```json
{
  "mcpServers": {
    "openkoutsi": {
      "url": "https://api.your-domain/mcp",
      "headers": { "Authorization": "Bearer okp_…" }
    }
  }
}
```

Admins can turn the endpoint off — see [ADMIN.md](ADMIN.md#turning-the-mcp-server-off).

## Evaluating LLM providers/models

[`llm-eval/`](llm-eval/) is a standalone [promptfoo](https://www.promptfoo.dev/) project for comparing models on the prompts the platform uses. See [llm-eval/README.md](llm-eval/README.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
