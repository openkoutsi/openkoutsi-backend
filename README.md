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
- **Inbox** — in-app messages notify athletes about their own events (earned achievements, personal access tokens about to expire or revoked by an admin) and admins about instance events (used invites); each user has an isolated per-user message store, deletions are permanent, and the design leaves a hook for future email/push delivery. Messages carry their own rendered `title` and `body`, written by `message_text` when the message is sent and stored on the row alongside the `locale` they were rendered in, so a new message type needs no matching template in the web app to be readable
- **Swappable email module** — a single, provider-agnostic seam (`backend/app/services/email/`) for all email: a generic `EmailProvider` interface (outbound `send`, inbound `verify_inbound_signature`/`parse_inbound`) with `LettermintProvider` and `EuromailProvider` (euromail.dev — EU-based, inbound included on its free tier) implementations, provider selection via `EMAIL_PROVIDER`, and self-rendered inline-styled HTML + plain-text bodies. Optional — with no provider configured, email-dependent features stay unavailable rather than erroring
- **Personal access tokens** — long-lived, scoped, revocable credentials a user issues to their own tooling (a backup script, a cron job, an external MCP client), so anything that isn't a browser can talk to the API for longer than the 60-minute access token allows. Format `okp_{id}_{secret}`; only the SHA-256 of the secret is stored and the token is shown once at creation. Scopes are per-resource read/write (`activities:read`, `plans:write`, …) and **default-deny** — a route reachable by a token has to say so, and a test walks every route to prove none was missed. A token can never reach the admin API (even when its owner is an admin), `/api/auth`, the inbox, the LLM endpoints, or the token endpoints themselves — so it can't mint another. Nor can it change the LLM *configuration* (`llm_base_url`, `llm_api_key`) or start a provider OAuth flow: both would let a scoped credential redirect where the user's own session sends data. `athlete:export` is a separate, audited grant rather than part of a general read. Name, scopes and expiry are fixed at creation; every token expires, with a one-year ceiling enforced server-side and no "never". Expiry is announced at 7 days, 1 day and on the day (inbox always, email opt-out per user) by a daily sweep, each stage exactly once. Revoked and expired rows are kept so a withdrawn credential stays distinguishable from an unknown one, tokens are revoked wholesale on password reset, and every token-authenticated request is written to a structured audit log keyed by token id — which is also the rate-limit key. Self-hosters can switch the whole feature off with `allow_personal_access_tokens`, which refuses *authentication*, not just issuance
- **MCP server** — a Model Context Protocol endpoint (`POST /mcp`, JSON-RPC 2.0) publishing **ten read-only, task-shaped coaching tools** over one athlete's own data: `get_training_status`, `find_activity`, `list_recent_activities`, `get_activity_detail`, `get_plan_status`, `get_goal_progress`, `get_power_profile`, `get_intensity_distribution`, `get_zone_totals`, `get_athlete_profile`. Instead of a prompt builder guessing what a coach needs and assembling a fixed context blob, the model asks. `get_training_status` answers for a **past date** as well as today (`as_of`), so "what shape was I in before that event" and "is this build steeper than the last one" are two comparable calls rather than guesswork — every figure moves with the date, the trailing volume window included, and a date before the athlete's recorded history is refused with the date that history starts on rather than answered with zeros a model would read as a collapse. `get_athlete_profile` is the one tool about the athlete rather than about training that happened: the power and heart-rate **zone boundaries** every zone figure from the other tools is measured against — nothing else published them, so a model handed "4 h 12 in Z2" either said nothing useful about it or invented the wattage — plus the physiology, the hours a week they say they have, and the coaching tone they asked for. It is deliberately not a profile dump: no name, no date of birth (an age in whole years instead), no FTP-test or weight history, no provider connections, no feature toggles and none of the BYOK model configuration — `athlete:export` is excluded from the callable scopes for exactly that reason and this must not become it by another door. It is also the only tool `athlete:read` opens on its own; before it, both tools declaring that scope also demanded `metrics:read`, so a token granted exactly the profile scope could call nothing. Two consumers reach the identical tools through different doors — an on-server agent, and external MCP clients authenticating with a personal access token — and both pass the same checks. Tools return **computed aggregates, never raw streams** (a three-hour ride holds ~11 000 samples per stream), preserve reason codes rather than flattening them to nulls (`too_short`, `no_power`, …), name the unit in every field description, bound every collection and report the true `total` alongside it, and never return coordinates. Failures come back as readable tool *results*, not exceptions — "No activity on 2026-07-14. Nearest rides: 2026-07-13 (endurance, 2 h 04) and 2026-07-16 (threshold, 1 h 12)." Access reuses the token scope vocabulary with **no `mcp:*` scope** (it would obscure the actual grant): each tool declares the read scopes it needs, a session credential holds all of them, and because the endpoint resolves its own credential (no single scope could be honest about ten differently-scoped tools) the route-policy walk never covers it, so the registry carries **its own default-deny** — a tool that declares nothing cannot be registered, and a test proves it. Consent is checked per invocation, calls are rate-limited per user (the in-process agent deliberately is not), and every invocation is audited with caller, tool, arguments, duration and outcome — never the result. **Genuinely read-only**: the two zone tools are explicitly asked not to freeze missing time-in-zone snapshots the way their REST counterparts do, and report the rides they couldn't count instead — a snapshot is permanent, so letting a `metrics:read` tool trigger one would let the moment an agent asked a question decide forever which zone definitions an old ride is judged against, and would make the `readOnlyHint` an MCP client uses to decide whether to prompt the user a lie. Self-hosters can switch the whole endpoint off with `allow_mcp_server` (default on), which refuses the handshake too
- **Admin dashboard** — manage users, invitations, password resets, an admin-contact shown on the password-reset page, instance-wide LLM settings, and list/revoke (never issue) a user's personal access tokens — metadata only, never their names, audited and announced in the user's inbox
- **FIT file ingestion** — upload activities directly with automatic Load, weighted power, and zone distribution analysis
- **Derived torque stream** — when an activity has both power and cadence, a per-second crank torque stream (Nm) is computed and stored alongside them, served from the activity streams endpoints; existing activities gain it on reprocess
- **Vectorised stream math** — power bests, distance bests, weighted power, torque, time-in-zone and aerobic decoupling are computed with numpy rather than per-sample Python loops, roughly 9× faster per activity (~75 ms → ~8 ms on a 3-hour 1 Hz ride). Invisible on a single upload; it is the paths that loop over history that gain — `POST /metrics/recalculate`, a provider backlog import, and zone-time backfills. The power–duration model fits (CP/W′, 3-parameter CP, exponential, power law) solve their whole parameter grid as one matrix; scipy was measured against them and rejected, because its bounded solvers turn a refused fit into a fabricated one
- **Aerobic response metrics** — every activity carries an **efficiency factor** (weighted power per heartbeat) and a **variability index** (weighted ÷ average power), both derived on read so they need no reprocess, plus **aerobic decoupling** — how far the power:heart-rate ratio drifted between the first and second half of the ride. Decoupling is only stored where it means something; otherwise the response carries a reason code instead of a misleading figure — `too_short` (under an hour, by elapsed time *and* sample count), `no_power` / `no_hr` (content-aware, so a paired-but-silent meter isn't reported as a heart-rate fault), `degenerate_hr`, `stream_mismatch` (the two recordings don't line up well enough to pair sample-for-sample), `variable_effort` (surging), or `uneven_pacing` (a ramp or negative split, which variability index alone doesn't catch). A per-second **W′ balance** stream (joules of anaerobic capacity remaining) is derived from the power stream against a CP/W′ snapshot fit from the athlete's power bests *as of that ride's date* and frozen on the activity, so a ride's W′ story doesn't change months later; existing activities gain it on reprocess. The fit is rejected outright unless it lands in a physiologically plausible range — the unconstrained least-squares intercept routinely produces a negative or near-zero W′ for a rider who only ever rides steady — and the stream is skipped on recordings too sparse to integrate as one sample per second. A provider history import walks newest-first while the fit only looks backwards in time, so during the walk every ride would be judged against a bests table holding nothing but newer rides — the snapshots are therefore **re-fit once the import finishes**, against the now-complete profile, still restricted to each ride's own as-of date. `cp_fit_points` records how many duration bests each fit had, so a thin one stays findable. `GET /api/metrics/efficiency` returns the efficiency trend across steady endurance rides, and the efficiency/decoupling figures are fed into the AI activity analysis
- **Manual activity entry** — log workouts by hand (date, duration, distance, avg/max HR, avg power, cadence, RPE/Load) with every field optional, behaving like a `manual` data provider
- **Workout categorization** — automatic Coggan-style zone classification with manual override
- **Strava + Wahoo sync** — OAuth integrations with history import and webhook updates through bridge services
- **Zone sync** — sync HR/power zones and FTP from connected providers
- **Accumulated time in zones** — `GET /api/metrics/zones/weekly` returns weekly (Monday-based) accumulated time in each power and HR zone across the selected period. Each activity's time-in-zone is captured as a frozen snapshot at processing time using the zones in effect then, so editing your zones only changes future activities and never rewrites past weeks
- **Intensity distribution** — `GET /api/metrics/intensity-distribution` collapses a training block (12 weeks by default) into three intensity bands — below LT1, between LT1 and LT2, above LT2 — and names the shape: polarized, pyramidal, threshold-heavy or predominantly low intensity. `method=time` sums the frozen zone snapshots, `method=session` counts each ride whole by its workout category; the two disagree by design, because warm-ups and coast-downs pull the time method toward pyramidal, so the method is always reported alongside the numbers. `basis` selects power or HR zones, preferring power. The response also carries coverage (how many rides in the window had usable data) and a flag when zone definitions moved inside the window, so a distribution drawn from a handful of rides can't pass for a confident one
- **Fixed zone model** — power zones are exactly 7 and HR zones exactly 5, validated on write (`PATCH /api/athlete`, and provider zone sync skips a non-conforming list rather than reshaping it). The three-band mapping reads zones positionally, so a variable-length list would silently change what a zone means
- **FTP estimation** — estimate FTP from your power curve via the 20-minute (95%) or Critical Power method, shown on the Power view, and accept either to set your profile FTP
- **Power curve (watts & W/kg)** — `GET /api/metrics/bests/power?metric=watts|wkg` returns the top-3 efforts per duration; the W/kg view ranks by watts-per-kg using the effective bodyweight at the time of each effort (snapshotted per power best when the activity is processed and never rewritten afterwards, so a new weigh-in applies from its own date onward and older efforts keep the weight they were ridden at — or none, if none was logged back then), so genuine W/kg PRs surface instead of just watts PRs divided by current weight
- **Power–duration models** — `GET /api/metrics/power-models` fits several models to your power curve (2- and 3-parameter Critical Power, a CP-anchored exponential, and a power law), returning each model's parameters, a sampled curve for plotting, the fit error, and your estimated potential (Neuromuscular Power/Pmax, Anaerobic Capacity, Maximal Aerobic Power and FTP) so modeled curves can be overlaid on the real power curve
- **Experience level** — self-reported athlete experience level (novice, intermediate, experienced, semi-pro, elite) stored on the profile via `PATCH /api/athlete` and fed into the LLM context for plan/workout generation and training-status, activity and goal analysis, so coaching and progression are tailored to the athlete's level
- **Fitness metrics** — Fitness/Fatigue/Form computed and shown as interactive charts; stale metrics caused by deleted activities are detected and corrected automatically on dashboard load. The fitness history card also shows cycling totals — number of activities, active time, and covered distance — for the selected time period
- **Fitness forecast** — `GET /api/metrics/fitness/forecast?days=90` projects Fitness/Fatigue/Form *forward* from the prescribed Load of your active plans, so you can see where your Form lands on a goal date and whether a plan ramps faster than intended before you ride it. The same Banister model as the historical series, seeded from today (metrics are caught up first, so the answer doesn't depend on whether the client hit `POST /metrics/catch-up` beforehand) and starting from tomorrow so measured days are never restated; loads from several active plans are summed, days with no prescribed workout decay as rest, and the projection continues past the end of the plan (the decaying tail is what detraining looks like). Computed on read — nothing is stored, so the forecast always reflects the plan as it stands now
- **Training calendar** — dashboard calendar shows both performed and planned workouts with distinct visual markers (completed, pending, skipped), and lets you mark a planned workout as done or skipped straight from the day view without opening the plan
- **Training plan generation** — periodized plans (Base → Build → Peak → Taper) with configurable **structure parameters**: a controlled week-over-week progression (5–10%, defaulting by experience level), a selectable build-to-recovery cadence (2–3 build weeks then a recovery week), a weekly **base Load** from non-workout riding (e.g. commuting, additive context), and an available weekly **training-hours range** that maps each week's total ride time into the athlete's availability. Each week carries **metadata** (build vs recovery, a focus note, target weekly Load/hours), and day descriptions are rendered to stay consistent with the prescribed duration and Load. Threshold/VO2max ("hard") days are capped per week and never scheduled back-to-back
- **Training plan editing** — edit plan metadata (name, goal, start date, length), edit/add/delete individual planned workouts from the calendar day view, and regenerate a plan's workouts (rule-based or AI); completed workouts are locked from edits and preserved on regeneration
- **Plan archiving/unarchiving** — creating a new plan only archives existing active plans whose dates overlap it, so plans covering different periods stay active together; archived plans can be reactivated via `POST /api/plans/{id}/unarchive`, which archives any overlapping active plan in turn
- **Activity → plan linking** — uploaded activities are automatically matched to the day's planned workout (sport, Load ≥ 60%, duration ≥ 60%); rest days are never auto-matched, and an activity already linked somewhere is left alone. Manual link/unlink via the plan calendar or the dashboard activity calendar; trying to link an activity that already belongs to another planned workout is refused with an error naming that workout's date, type and plan. A single session that was recorded as **several activities** (an accidental stop, a coffee break, back-to-back virtual rides) can have each part linked to the same planned workout, so their combined duration and Load together satisfy the goal — auto-matching still links one activity, the extras are linked by hand
- **Workout skip tracking** — mark planned workouts as skipped with a reason (illness, injury, fatigue, travel, weather, etc.) for accurate training log and LLM coaching context
- **Plan adherence score** — two deterministic, always-on scores (not gated behind the LLM subscription): a per-workout **match score** (0–100) grading how well the linked activities hit the target Load and duration — over- and under-performing both hurt, though a completed workout never scores below 50 (doing the session, however far off, always beats missing it) — and a Load-weighted **plan adherence score** (0–100) rolling those up over the elapsed portion of a plan, with missed sessions counting as zero and skips softened by reason. Computed continuously (webhook/manual ingest and first read of the day) and persisted as a daily snapshot per active plan for charting via `GET /plans/{id}/adherence`; the snapshot series self-heals — stored days made stale by retroactive changes (a link/unlink to an old workout, an edited past workout, a formula change) are rewritten on the next recompute. The plan's live adherence summary also reports the count of **remaining sessions** still to do from today onward (future workouts plus today's un-acted one)
- **Structured workouts** — create interval workouts and export as Zwift `.zwo` or FIT workout files for head units (FIT export flattens repeat blocks into individual consecutive steps for reliable display on Wahoo/Garmin devices)
- **Push workouts to Wahoo** — send a structured workout straight to a connected Wahoo account as a plan + scheduled workout, so it appears in Planned Workouts on ELEMNT/RIVAL (schedule within today→+6 days; re-pushing updates instead of duplicating)
- **Generate workouts from a plan** — auto-synthesize structured interval workouts (via LLM) for a training plan's upcoming days in one action; generated workouts are cached on the planned workout (already-generated days are skipped, so no extra LLM calls), rest/out-of-window days are skipped, and a per-day result summary shows what was generated, skipped, or failed. The generated workouts appear in the Workouts tab, where you can review, edit, and upload them to Wahoo individually
- **Goals** — set training/event goals with optional target metrics and dates; when marking a goal achieved, record the final achieved value and a free-text outcome note capturing whether the target was reached
- **AI goal guidance** — on demand, an LLM judges how realistic a goal is for its timeline given the athlete's current fitness and trend (a `realistic`/`ambitious`/`unrealistic` verdict) and gives concrete coaching steps to reach it; streamed in the Koutsi coach voice and persisted per goal
- **Activity labels & notes** — tag activities as "race" or "commute" and add free-text notes (included in AI analysis context); the activity list can filter by label, e.g. show only races (`?labels=race`) or hide commutes (`?exclude_labels=commute`)
- **RPE (perceived effort)** — record a subjective 1–10 effort score on any activity (settable via `PATCH /api/activities/{id}`, persisted from manual entry, and fed into the AI analysis alongside measured intensity); after a significant cycling ride lands the athlete is prompted to rate it, driven by a server-side `rpe-queue` (`GET /api/activities/rpe-queue`) with a `rpe_head` cursor in `app_settings`, toggleable per-athlete via `app_settings.ask_for_rpe`
- **Achievements & streaks** — deterministic, always-on gamification (not gated behind the LLM subscription): tiered badges for volume, distance, climbing (including `everesting` — 8 848 m in a single ride), variety, engagement, finished plans and reached goals, plus weekly streaks for active weeks, hours, distance, climbing and distinct sports, and a monthly active streak. **Weekly granularity only — no daily streaks**, deliberately, since forcing a ride every single day over long periods isn't healthy to reward; the current week is always "in progress" and never breaks a streak before it ends. Unlocks are derived state like Fitness/Fatigue/Form: recomputed idempotently on every ingest and on the daily first read, dated by *when the criterion was actually met* (back-filling old rides ages a badge instead of re-dating it to today), and revoked when the underlying activity goes away. Badges whose data an athlete can't produce (elevation without a barometric FIT, Load without power or HR) are hidden rather than shown permanently locked. Served by `GET /api/achievements` and `GET /api/achievements/streaks`, announced in the inbox (one message per batch, never one per tier — and the message names every badge in the batch rather than just counting them), and toggleable per-athlete via `app_settings.gamification` — offered in the onboarding wizard as well as in settings
- **AI coaching analysis** — per-activity analysis and plan support with OpenAI-compatible backends; cycling activities get a detailed coaching breakdown while other sports are treated as supplemental training and receive a short, encouraging acknowledgement. When an activity is linked to a planned workout, that planned workout is included in the analysis context so the coach can comment on plan adherence
- **Koutsi daily feedback** — dashboard card with LLM-generated daily training status covering load trends, recovery state, plan adherence (across all active plans — current ones with this week's workouts, plus a note for any upcoming plan), and goal progress; auto-triggers after uploads/syncs when enabled
- **Agentic Koutsi** — opt-in (`app_settings.agentic_koutsi`), the daily status card and per-activity analysis stop being a fixed context blob assembled ahead of time and become an agent loop over the MCP tools: Koutsi decides what it needs and asks, so it can follow a thread — see a flat form number, go and look at the last four weeks, and say *why*. Because the tool rounds emit no prose, the run persists a **progress code** (`thinking`, `tool.get_power_profile`, …) in its own column on the same 500 ms cadence as the text, which the web app localises; a code, not model prose, so it translates across all fourteen languages and can't leak tool internals into the card. It's cleared the moment the answer starts, so a finished card looks exactly as it always did. Everything degrades to the single-shot prompt at runtime rather than failing — that path is not legacy and stays tested, because under BYOK the hoster doesn't control whether the athlete's model can call tools. A provider that rejects the `tools` param (detected like `response_format`), a preset flagged `"tools_supported": false`, a model that accepts `tools` and then calls none, a run that exhausts the round cap (6 for the status card, 3 for one activity) or the 24 000-character tool-result budget, a process already running `AGENT_MAX_CONCURRENT_RUNS` loops, and **any other upstream failure before prose has been written** — a 429, a 5xx, a dropped connection, and above all a context-length 400, which is a failure the loop itself creates and the small windows on self-hosted models make likely. The one thing that does *not* degrade is a provider saying our own function schema is invalid: that's our bug, and hiding it behind a quietly worse answer for everyone is the failure the rule exists to prevent. Bulk provider backlog imports always take the blob path: 4–6× the calls per activity is a real bill on the one path nobody reads one analysis at a time. Tool failures come back as sentences the model can act on, results are truncated with an explicit marker rather than silently, a turn is held to 4 tool calls (with a sentence for the rest, so the model knows to ask again rather than reason from a gap), the instance's `llm_analysis_context` is resent on every turn, the `MOOD:` format rule is restated on **every turn that follows tool results** — not just the last one, since models obey a leading-format rule less reliably after tool results and the turn that answers is usually the second or third, not the capped one — and token usage is **summed across every call in the run** so the admin usage summary isn't under-reporting agentic analyses by the number of turns they took. The tools reckon dates from the *athlete's* timezone, not the server's, so "not due yet" and "missed" agree with the date the prompt asserts
- **Conversational Koutsi** — a dedicated chat page (rides the same `app_settings.agentic_koutsi` opt-in, since a chat that can't look at the athlete's data is a general-purpose LLM in a cycling hat) where the athlete asks their own questions and Koutsi answers them by calling the MCP tools. Messages are built **server-side** — the client sends one string, never a message array, which is the whole difference from the `/api/llm/chat` proxy #45 removed and the reason the scope policy in the system prompt is not something a token holder can edit out. That policy is four bands: coaching questions answered fully, adjacent ones (fuelling, sleep, strength, bike fit) answered as a coach rather than refused — refusal theatre is a bug here, not a safe default — medical ones redirected to a clinician without ever diagnosing or advising training through symptoms, and unrelated ones declined in a sentence. It is restated on every turn, including the ones after tool results where #43 measured leading-format rules degrade, and BYOK caps what any of it can promise: with your own model the output is your model's. Only the dialogue is stored — tool calls and results are deliberately not, because they are most of the bytes, they go stale, and re-running a read-only tool on a later turn is *more* correct than replaying its old answer; what survives is `tool_names`, written through on every progress marker rather than when the turn settles, so the web app can draw each lookup as a step where it happened instead of listing them all under a finished answer. Conversations are deletable per-thread and land in the GDPR export as `chat.json`. Unlike every other LLM surface chat has **no single-shot prompt to fall back on**, so each failure is visible and typed rather than silently degraded: a turn that can't get one of the `AGENT_MAX_CONCURRENT_RUNS` slots *queues* (visibly) instead of being refused, a model that can't call tools disables the surface up front rather than failing after the athlete has composed a question, and upstream errors settle the turn with a code the web app localises. Per-day and per-conversation turn budgets bound the first LLM surface an athlete can trigger arbitrarily often
- **Stranded-run recovery** — the three surfaces that write a `pending` LLM status (training status, goal guidance, activity analysis) are judged alive by an **inactivity** clock rather than a start time: the timestamp is touched on every progress commit, so a slow-but-healthy stream is never declared dead underneath itself, and one that has gone quiet for 15 minutes is released. Activity analyses carry their own `analysis_updated_at` for this, and gain the age check the other two always had — without it a `pending` row was terminal, since `POST /api/activities/{id}/analyze` refuses while one is in flight, so that ride could never be analysed again. Nothing that sets `pending` survives the process (`BackgroundTasks` and `asyncio.create_task` both die with it), so an ordinary redeploy strands whatever was mid-run; startup therefore settles every `pending` row before serving its first request, across all per-user databases
- **AI transparency** — every response carrying model-written prose marks it as such: `analysis_ai_generated` on the activity detail, `feedback_ai_generated` on the training status, and `guidance_ai_generated` on goal guidance. Each is derived from its prose field, so the flag is true exactly when there is generated text to disclose. The web frontend renders a matching visible notice under the coaching text (EU AI Act Article 50 transparency)
- **API v2** — token-scoped (no team slug in any path), no trailing slashes on collection roots, a shared pagination envelope across all collections (activities, goals, plans, workouts, messages, admin lists), analytics consolidated under `/api/metrics`, and `PATCH /api/athlete` for partial updates
- **Privacy-first** — explicit GDPR consent for health-data processing (enforced server-side on the ingestion paths: provider connect and manual upload), a configurable privacy-policy link (`PRIVACY_POLICY_URL`, default `koutsi.dev/privacy`) surfaced on the consent screen, and export/delete your data at any time (the export is a complete per-user dump — profile & LLM settings, activities with notes/labels/RPE/analysis, plans, goals, structured workouts, daily fitness metrics, personal records, achievements, inbox, weight log, personal access tokens, and raw FIT files)
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

Cross-cutting plumbing lives in one place per concern, so route handlers and
services stay focused on what is actually different between them:

| Concern | Home |
|---|---|
| Auth + per-user session + athlete lookup for a route | `core/deps.py` (`get_ctx_session_athlete`) |
| Plan / planned-workout ownership checks | `api/plans.py` (`get_owned_plan`, `get_owned_workout`) |
| Streaming LLM analyses (transport, DB drain loop, usage recording) | `services/llm_streaming.py` |
| Non-streaming LLM calls (config resolution, structured outputs) | `services/llm_client.py` |
| Turning imported activity data into stored metrics, bests and intervals | `services/provider_sync.py` |
| What an AI coach may ask for, and what it gets back | `mcp/registry.py` (declarations) + `mcp/dispatch.py` (every check) |

## Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12 · FastAPI · SQLAlchemy 2 (async) · Alembic |
| Database | SQLite (WAL mode) |
| Auth | JWT (`python-jose` · `passlib`) |
| FIT parsing | fitdecode |
| Stream & fit math | numpy |
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

**Tool calling for the agentic coach** follows the same three-part shape, because
it has the same problem: under BYOK the hoster does not control whether the
athlete's server can do function calling, and support across that population
ranges from good to absent to present-but-wrong. So `tools` is sent by default; a
provider that rejects the *parameter* is detected at runtime and the run falls
back to the single-shot blob prompt; and a preset can pre-empt that with
`tools_supported: false` — which is the flag that matters for the nastier case, a
server that accepts `tools` and then emits nonsense, since that one never
produces the 400 runtime detection needs. Neither detector swallows an
"invalid schema" body: that means *our* pydantic model is broken, and degrading
every provider silently would hide it.

Admins configure, per instance (Settings → AI / LLM), a **list of selectable
presets** — each a self-contained connection: display name, stable identifier,
base URL, model id, API key, headers and extra chat-completion body params (e.g.
`max_tokens` or a thinking/`reasoning_effort` config). This lets an admin offer
distinct providers (Anthropic, Mistral, …) as presets. **The first preset in the
list is the instance default.** Users pick a preset — the dropdown shows each
preset's display name, but the selection is stored by its stable identifier, so
renaming a display name never breaks existing selections. A user's selected
preset (or BYOK server) is honoured everywhere an LLM is called on their behalf —
the plan/workout generators **and** the background analysers (activity analysis,
training status, goal guidance) — falling back to the instance default only when
they haven't chosen one.

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

## MCP server

`POST /mcp` speaks the Model Context Protocol (revision `2025-06-18`) over
JSON-RPC 2.0, stateless: `initialize`, `notifications/initialized`, `ping`,
`tools/list` and `tools/call`. Tools only — resources and prompts are absent from
the advertised capabilities rather than stubbed, so a client never offers a user
something that will fail.

Point any MCP client at it with a personal access token:

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

The token's scopes decide which tools answer. `tools/list` returns every tool
regardless — hiding the unreachable ones would make a scope refusal look like a
missing feature — with the scopes each needs in its `_meta`, so a client can
explain the gap rather than discover it by failing. All ten tools are covered by
the five read scopes `activities:read`, `athlete:read`, `goals:read`,
`metrics:read` and `plans:read`, and each of those five opens at least one tool
on its own; `athlete:export` is deliberately **not** callable, since one call
returning the whole record is the opposite of task-shaped — which is also why
`get_athlete_profile` publishes the coaching-relevant settings and zone
boundaries rather than the profile record.

An admin can turn the endpoint off for the whole instance with `allow_mcp_server`
— the **Allow the MCP server** switch in the admin console's Settings tab,
default **on** — which refuses it outright — handshake included —
rather than letting a client connect to a server that will decline every useful
call. Denying `/mcp` at the reverse proxy works too. Either way it narrows the
*interface*, not the exposure: the same data is reachable through the ordinary
REST routes with the same token, and what limits a credential is its scopes. See
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
