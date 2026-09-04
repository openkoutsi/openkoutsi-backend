# openkoutsi-backend

[![codecov](https://codecov.io/gh/openkoutsi/openkoutsi-backend/graph/badge.svg)](https://codecov.io/gh/openkoutsi/openkoutsi-backend)

The backend (FastAPI API + bridge services + core library) for openkoutsi, a self-hosted cycling coaching platform. Upload FIT files or sync from Strava/Wahoo, track fitness metrics (Fitness/Fatigue/Form), and generate periodized training plans from your own server.

> **koutsi** (κουτσί) — Finnish for "coach"

> **Web frontend:** the Next.js UI lives in a separate repository, [openkoutsi/openkoutsi-web](https://github.com/openkoutsi/openkoutsi-web).

## Why

Most cycling coaching tools are cloud-only SaaS. openkoutsi is different: you run it on your own hardware, your data stays under your control, and integrations are optional.

## Features

### Accounts and access

- **Single instance, per-user data** — one deployment; every user's athlete profile and all training data live in their own isolated SQLite database
- **Signup** — the setup wizard creates the first administrator; further accounts come from admin invites, or from self-serve email signup (register → verify → activate) when `allow_self_signup` is on and an email provider is configured. The setup endpoint settles the "first admin" check inside the write (`INSERT … SELECT … WHERE NOT EXISTS`), so concurrent requests cannot both create an administrator — the second is told 409
- **Changing the account's email address** — `POST /api/auth/change-email` moves an account to a new address, or sets one for the first time on an invite-created account. Both ends must approve: hashed, single-use, 24-hour links go to the new address *and* to the one being left, and nothing moves until both are opened. The address is the account's only self-serve root of trust (passwords are set through reset tokens mailed to `users.email`), so a one-sided change would let a password holder relocate that channel and take the account permanently. Changing also costs the current password; a first-time set needs only the new side. The response is a fixed acknowledgement whether the target is free, taken or the caller's own, so it cannot be used to probe who has an account. Sessions survive. A password reset cancels any change in flight. An abandoned, never-verified signup does not reserve an address
- **Admin recovery for an unreachable address** — `PATCH /api/admin/users/{id}/email` sets or clears an account's address, for a user whose old mailbox is gone. It assumes the account may be in the wrong hands: every session and personal access token is withdrawn and any change in flight is voided
- **Confirmed-address visibility** — `GET /api/admin/users` returns `email_verified_at` beside `email`, shown per account in the console's Users tab. Login by email requires that stamp, so an unfinished self-serve signup is a row that can never sign in. Null on an account with no address means "nothing to confirm", not "unconfirmed"
- **Self-serve password reset** — with email configured, users request a reset link from "Forgot password?" (`POST /api/auth/request-password-reset`); admins can still mint links directly. Both are single-use and expire in 1 hour
- **Session invalidation** — every access and refresh token carries the `users.token_version` generation counter current when it was minted, checked against the row on every request. Raising it ends every open session, which is what a password reset does, and what `POST /api/auth/logout-all` does on demand. Plain `POST /api/auth/logout` still clears this browser's cookie only
- **Password length is validated, not discovered** — bcrypt hashes at most 72 bytes and raises rather than truncating. Every password field enforces the limit in *bytes* (72 emoji are 288) and returns a 422 that says so. Fields that only check a password (login, delete-account) enforce length alone
- **No account enumeration, on any channel** — signup and password reset return fixed acknowledgements; login verifies against a fixed dummy bcrypt hash when no account matches, so an unknown identifier costs the same as a known one (measured 66× apart before, ~1× after)
- **Secrets are encrypted, or you said otherwise out loud** — `ENCRYPTION_KEY` is required just as `SECRET_KEY` is. The backend refuses to start without a key or with one Fernet cannot load, and accepts `ALLOW_PLAINTEXT_SECRETS=true` as a deliberate, warned-on-every-start choice for throwaway instances. An unreadable secret reads as absent, never as itself: a value that was never a Fernet token passes through (the pre-encryption migration path), while ciphertext this key cannot open reads as `None` with an error naming `ENCRYPTION_KEY` — `None` rather than an exception, since raising would abort the whole result set
- **Personal access tokens** — long-lived, scoped, revocable credentials a user issues to their own tooling (backup scripts, cron jobs, external MCP clients). Format `okp_{id}_{secret}`; only the SHA-256 of the secret is stored and the token is shown once. Scopes are per-resource read/write (`activities:read`, `plans:write`, …) and **default-deny** — a route reachable by a token must say so, and a test walks every route to prove none was missed. A token can never reach the admin API (even for an admin owner), `/api/auth`, the inbox, the LLM endpoints or the token endpoints themselves, nor change the LLM configuration or start a provider OAuth flow. `athlete:export` is a separate, audited grant. Name, scopes and expiry are fixed at creation; every token expires, with a one-year server-side ceiling. Expiry is announced at 7 days, 1 day and on the day (inbox always, email opt-out per user) by a daily sweep, each stage once. Revoked and expired rows are kept, tokens are revoked wholesale on password reset, and every token-authenticated request is audited by token id — which is also the rate-limit key. `allow_personal_access_tokens` switches the feature off, refusing *authentication* rather than just issuance
- **Admin dashboard** — manage users, invitations, password resets, an admin-contact shown on the password-reset page, instance-wide LLM settings, and list/revoke (never issue) a user's personal access tokens — metadata only, never their names, audited and announced in the user's inbox
- **Privacy-first** — explicit GDPR consent for health-data processing (enforced server-side on provider connect and manual upload), a configurable privacy-policy link (`PRIVACY_POLICY_URL`, default `koutsi.dev/privacy`), and export/delete your data at any time. The export is a complete per-user dump: profile & LLM settings, activities with notes/labels/RPE/analysis, plans, goals, structured workouts, daily fitness metrics, personal records, achievements, inbox, weight log, personal access tokens and raw FIT files
- **AI transparency** — every response carrying model-written prose marks it: `analysis_ai_generated` on the activity detail, `feedback_ai_generated` on the training status, `guidance_ai_generated` on goal guidance. Each is derived from its prose field, so the flag is true exactly when there is generated text to disclose. The frontend renders a matching visible notice (EU AI Act Article 50)
- **API v2** — token-scoped (no team slug in any path), no trailing slashes on collection roots, a shared pagination envelope across all collections, analytics consolidated under `/api/metrics`, and `PATCH /api/athlete` for partial updates

### Ingestion and activity data

- **FIT file ingestion** — upload activities directly with automatic Load, weighted power and zone distribution analysis
- **Bulk import (FIT / GPX / TCX, gzipped, or in a zip)** — `POST /api/activities/import` takes many files or a whole Strava bulk export archive and works through it as a background job with a pollable status resource (`GET /api/activities/imports[/{id}]`). The archive is walked recursively with the guards a file from elsewhere needs: caps on entry count, per-entry and total *uncompressed* size, bounded nesting depth, and rejection of traversing entry names. GPX and TCX are parsed into the same `Profile` the FIT parser produces, so Load, weighted power, zone snapshots, power bests, torque and interval extraction work on them unchanged. Duplicates are collapsed within the batch as well as against existing activities, keeping the richest copy when an export holds one ride as FIT *and* TCX *and* GPX. Every file gets a per-file outcome with a reason, so one corrupt file costs one ride rather than the job. Originals are stored in the format they arrived in (`ActivitySource.format`); download and reprocess dispatch on it
- **Manual activity entry** — log workouts by hand (date, duration, distance, avg/max HR, avg power, cadence, RPE/Load) with every field optional, behaving like a `manual` data provider
- **No location data from a ride, in any format** — the GPX parser reads coordinates, derives distance and elevation gain, and drops them before returning: `Profile` has no field that could carry a coordinate and `ActivityStream` has no channel for one. `openkoutsi.gpx.extract_route()` is the deliberate, separately-named exception and is never called on the ingestion path. Route geometry is kept only for a **course** the athlete uploads on purpose
- **Derived distance is judged by speed, not by metres** — GPX and TCX carry no distance of their own, so it is summed from coordinates. A step counts as travel when its *implied speed* is plausible (`geo.step_is_travel`, 216 km/h), which is the only test that holds at every recording rate; the old flat 500 m per-step cap assumed ~1 Hz recording and silently deleted real distance on variable-rate tracks. A track with no clock at all (a course file) falls back to a metre cap set where a teleport begins. Genuine glitches are still rejected
- **Streams on a common clock** — every stream is a 1 Hz series indexed by *second*, not by sample: index `i` is second `i` of the ride in every channel, with `null` where that channel had nothing to record. Both ingest paths resample onto this grid (the FIT parser, and Strava's arrays via its `time` stream), so `w_bal`'s joules-per-second arithmetic is exact, a lap's window covers the seconds it claims to, and a gap is time in no zone rather than time in Z1. A gap is never a zero: metrics about the *rider* (average power, power bests, weighted power) read past it, while metrics about the *clock* (distance bests, W′ balance) count it. `stream_mismatch` is a measured overlap between two channels, not a difference in list length. Nothing rewrites an activity's streams after ingestion, so rides stored before this keep the old dense shape until re-ingested; every consumer reads both shapes
- **Derived torque stream** — when an activity has both power and cadence, a per-second crank torque stream (Nm) is computed and stored alongside them, served from the activity streams endpoints; existing activities gain it on reprocess
- **Vectorised stream math** — power bests, distance bests, weighted power, torque, time-in-zone and aerobic decoupling are computed with numpy rather than per-sample Python loops, roughly 9× faster per activity (~75 ms → ~8 ms on a 3-hour 1 Hz ride). The paths that gain are the ones that loop over history: `POST /metrics/recalculate`, provider backlog imports, zone-time backfills. The power–duration model fits solve their whole parameter grid as one matrix; scipy was measured and rejected, because its bounded solvers turn a refused fit into a fabricated one
- **Activity parsing stays off the event loop** — reading a FIT file is pure-Python iteration over the whole file (~11 s for a 4.8 MB ride, under a tenth of the 50 MB upload limit). Every parse runs in a worker thread (`asyncio.to_thread`): the upload handler's start-time read, the background processing after it, the provider-sync FIT summary, the interval rebuild and the Wahoo stream extraction. Starlette runs `BackgroundTasks` *on* the event loop, so the background half stalled the process as thoroughly as the inline half. A heartbeat test asserts the loop keeps ticking through each path
- **Activity labels & notes** — tag activities as "race" or "commute" and add free-text notes (included in AI analysis context); the activity list can filter by label (`?labels=race`, `?exclude_labels=commute`)
- **Commute detection** — write rules describing what your commute looks like (sport types, distance and duration bands, local time-of-day windows, days of week, in `app_settings.commute_rules`) and openkoutsi *suggests* the `commute` label for you to confirm — it never applies a guess, because the `commuter` badge counts labelled rides and the RPE prompt skips them. Strava's own commute flag *is* applied directly: that one is your assertion, not a heuristic. Rules can be proposed from commutes you already labelled (`GET /api/activities/commute/proposal`, ten-ride minimum), the back catalogue scanned on request (`POST /api/activities/commute/scan`), and pending rides listed with `?suggested_label=commute`. Editing a rule re-evaluates everything unanswered; a dismissal is permanent and survives reprocessing. No GPS is involved
- **RPE (perceived effort)** — record a subjective 1–10 effort score on any activity (via `PATCH /api/activities/{id}`, persisted from manual entry, fed into AI analysis alongside measured intensity). After a significant cycling ride lands the athlete is prompted to rate it, driven by a server-side `rpe-queue` (`GET /api/activities/rpe-queue`) with a `rpe_head` cursor in `app_settings`, toggleable via `app_settings.ask_for_rpe`

### Metrics and analysis

- **Fitness metrics** — Fitness/Fatigue/Form computed and shown as interactive charts; stale metrics caused by deleted activities are detected and corrected on dashboard load. The fitness history card also shows cycling totals (activity count, active time, distance) for the selected period
- **Fitness forecast** — `GET /api/metrics/fitness/forecast?days=90` projects Fitness/Fatigue/Form *forward* from the prescribed Load of active plans, so you can see where Form lands on a goal date and whether a plan ramps faster than intended. Same Banister model as the historical series, seeded from today (metrics are caught up first) and starting from tomorrow so measured days are never restated. Loads from several active plans are summed, days with no prescribed workout decay as rest, and the projection continues past the end of the plan. Computed on read — nothing is stored
- **Workout categorization** — automatic Coggan-style zone classification with manual override
- **Accumulated time in zones** — `GET /api/metrics/zones/weekly` returns weekly (Monday-based) accumulated time in each power and HR zone. Each activity's time-in-zone is a frozen snapshot taken at processing time with the zones in effect then, so editing your zones only affects future activities
- **Intensity distribution** — `GET /api/metrics/intensity-distribution` collapses a training block (12 weeks by default) into three bands — below LT1, LT1–LT2, above LT2 — and names the shape: polarized, pyramidal, threshold-heavy or predominantly low intensity. `method=time` sums the frozen zone snapshots, `method=session` counts each ride whole by its workout category; the two disagree by design (warm-ups and coast-downs pull the time method toward pyramidal), so the method is always reported alongside the numbers. `basis` selects power or HR zones, preferring power. The response carries coverage and a flag when zone definitions moved inside the window
- **Fixed zone model** — power zones are exactly 7 and HR zones exactly 5, validated on write (`PATCH /api/athlete`; provider zone sync skips a non-conforming list rather than reshaping it). The three-band mapping reads zones positionally, so a variable-length list would silently change what a zone means
- **FTP estimation** — estimate FTP from your power curve via the 20-minute (95%) or Critical Power method, shown on the Power view; accept either to set your profile FTP
- **Power curve (watts & W/kg)** — `GET /api/metrics/bests/power?metric=watts|wkg` returns the top-3 efforts per duration. The W/kg view uses the effective bodyweight at the time of each effort (snapshotted per power best at processing time and never rewritten), so a new weigh-in applies from its own date onward and genuine W/kg PRs surface instead of watts PRs divided by current weight
- **Power–duration models** — `GET /api/metrics/power-models` fits several models to your power curve (2- and 3-parameter Critical Power, a CP-anchored exponential, a power law), returning each model's parameters, a sampled curve for plotting, the fit error, and estimated potential (Neuromuscular Power/Pmax, Anaerobic Capacity, Maximal Aerobic Power, FTP) for overlaying on the real curve
- **Aerobic response metrics** — every activity carries an **efficiency factor** (weighted power per heartbeat) and a **variability index** (weighted ÷ average power), both derived on read so they need no reprocess, plus **aerobic decoupling** — how far the power:heart-rate ratio drifted between the halves of a ride. Decoupling is stored only where it means something; otherwise the response carries a reason code rather than a misleading figure: `too_short` (under an hour, by elapsed time *and* sample count), `no_power` / `no_hr` (content-aware, so a paired-but-silent meter isn't reported as a heart-rate fault), `degenerate_hr`, `stream_mismatch`, `variable_effort` (surging) or `uneven_pacing` (a ramp or negative split, which variability index alone misses). A per-second **W′ balance** stream (joules of anaerobic capacity remaining) is derived from the power stream against a CP/W′ snapshot fit from the athlete's power bests *as of that ride's date* and frozen on the activity, so a ride's W′ story doesn't change months later; existing activities gain it on reprocess. The fit is rejected unless it lands in a physiologically plausible range, and the stream is skipped on recordings too sparse to integrate per second. A provider history import walks newest-first while the fit only looks backwards, so snapshots are **re-fit once the import finishes** against the now-complete profile, still restricted to each ride's own as-of date. `cp_fit_points` records how many duration bests each fit had. `GET /api/metrics/efficiency` returns the efficiency trend across steady endurance rides, and the efficiency/decoupling figures feed the AI activity analysis
- **Experience level** — self-reported (novice, intermediate, experienced, semi-pro, elite) via `PATCH /api/athlete`, fed into the LLM context for plan/workout generation and training-status, activity and goal analysis
- **Achievements & streaks** — deterministic, always-on gamification (not gated behind the LLM subscription): tiered badges for volume, distance, climbing (including `everesting` — 8 848 m in a single ride), variety, engagement, finished plans and reached goals, plus weekly streaks for active weeks, hours, distance, climbing and distinct sports, and a monthly active streak. **Weekly granularity only — no daily streaks**, deliberately: the current week is always "in progress" and never breaks a streak before it ends. Unlocks are derived state like Fitness/Fatigue/Form — idempotently reconciled, dated by when the criterion was actually met (back-filling old rides ages a badge rather than re-dating it to today), and revoked when the underlying activity goes away. There is no incremental path, so **writes only mark the athlete and reads settle**: an upload, edit or sync stamps `achievements_dirty_at` and returns, and the reconcile runs on the next achievements read, the daily first read, the data export, or a daily background sweep. Deferring costs only latency, since unlocks are a pure function of the data, and it turns importing a season into N cheap marks rather than N full-history scans. Badges whose data an athlete can't produce (elevation without a barometric FIT, Load without power or HR) are hidden rather than shown permanently locked. Served by `GET /api/achievements` and `GET /api/achievements/streaks`, announced in the inbox (one message per batch, naming every badge in it), toggleable via `app_settings.gamification`

### Planning

- **Training calendar** — the dashboard calendar shows both performed and planned workouts with distinct markers (completed, pending, skipped), and lets you mark a planned workout done or skipped from the day view
- **Training plan generation** — periodized plans (Base → Build → Peak → Taper) with configurable structure parameters: a controlled week-over-week progression (5–10%, defaulting by experience level), a selectable build-to-recovery cadence (2–3 build weeks then a recovery week), a weekly base Load from non-workout riding, and an available weekly training-hours range. Each week carries metadata (build vs recovery, a focus note, target weekly Load/hours), and day descriptions are rendered to stay consistent with the prescribed duration and Load. Hard (threshold/VO2max) days are capped per week and never scheduled back-to-back
- **Training plan editing** — edit plan metadata (name, goal, start date, length), edit/add/delete individual planned workouts from the calendar day view, and regenerate a plan's workouts (rule-based or AI); completed workouts are locked from edits and preserved on regeneration
- **Plan archiving/unarchiving** — creating a plan only archives existing active plans whose dates overlap it, so plans covering different periods stay active together. Archived plans reactivate via `POST /api/plans/{id}/unarchive`, which archives any overlapping active plan in turn
- **Activity → plan linking** — uploaded activities are automatically matched to the day's planned workout (sport, Load ≥ 60%, duration ≥ 60%); rest days are never auto-matched, and an activity already linked elsewhere is left alone. Manual link/unlink from the plan or dashboard calendar; linking an activity that belongs to another planned workout is refused with an error naming that workout's date, type and plan. A single session recorded as several activities can have each part linked to the same planned workout, so their combined duration and Load satisfy the goal — auto-matching still links one, the extras by hand
- **Workout skip tracking** — mark planned workouts skipped with a reason (illness, injury, fatigue, travel, weather, …) for an accurate training log and LLM coaching context
- **Plan adherence score** — two deterministic, always-on scores: a per-workout **match score** (0–100) grading how well the linked activities hit target Load and duration — over- and under-performing both hurt, though a completed workout never scores below 50 — and a Load-weighted **plan adherence score** (0–100) rolling those up over the elapsed portion of a plan, with missed sessions counting zero and skips softened by reason. Computed continuously (webhook/manual ingest and the first read of the day) and persisted as a daily snapshot per active plan for charting via `GET /plans/{id}/adherence`. The series self-heals: stored days made stale by retroactive changes are rewritten on the next recompute. The live adherence summary also reports **remaining sessions** from today onward
- **Structured workouts** — create interval workouts and export as Zwift `.zwo` or FIT workout files (FIT export flattens repeat blocks into consecutive steps for reliable display on Wahoo/Garmin devices)
- **Push workouts to Wahoo** — send a structured workout to a connected Wahoo account as a plan + scheduled workout, appearing in Planned Workouts on ELEMNT/RIVAL (schedule within today→+6 days; re-pushing updates instead of duplicating)
- **Generate workouts from a plan** — auto-synthesize structured interval workouts (via LLM) for a plan's upcoming days in one action. Generated workouts are cached on the planned workout (already-generated days are skipped, so no extra LLM calls), rest/out-of-window days are skipped, and a per-day summary shows what was generated, skipped or failed. They appear in the Workouts tab for review, editing and individual upload to Wahoo
- **Goals** — set training/event goals with optional target metrics and dates; when marking a goal achieved, record the final achieved value and a free-text outcome note

### Courses

- **Course recon — a GPX course becomes a pacing plan** — `POST /api/courses` takes a course file and returns a segment table: the route thinned to ~8 m spacing, elevation smoothed over a **distance** window before any gradient is derived (differencing raw samples swings gradients tens of percent between points, the single biggest correctness risk in the feature), split where gradient meaningfully changes, with runs too short to pace dissolved into their neighbours. Each segment carries a power target and a predicted split solved from the steady-state balance `P·η = v·(Crr·m·g·cosθ + m·g·sinθ) + ½·ρ·CdA·v³`, using the athlete's own FTP and weight plus a **bike** (`/api/bikes`: tyre width → Crr, riding position → CdA) — deterministic arithmetic in `openkoutsi/course.py`, solved by bisection rather than a continuous optimiser.

  Ask for a **target time** and the inverse solver distributes effort across the segments to hit it, or refuses with a reason code rather than returning a number that needs 140% of FTP for four hours — `target_faster_than_physics`, or `exceeds_sustainable_power` with the intensity it would take against what is sustainable for that duration. A refusal is a result, not an error: the athlete still gets the course. Name a **target power** instead and the mirror solver holds it as the ride's *average* (the gradient weighting still spends on the climbs) and reports the finish time; an unsustainable power is flagged the same way but **keeps its splits**, because unlike an impossible time it still describes a ride the model can lay out. Either target can be set, swapped or cleared at any time, which re-solves from the stored track.

  The written plan — fuelling, decision points, where the day is won — comes from the same streamed-prose machinery as goal guidance, given the **derived table and never the track**, told to do no arithmetic, and required to say plainly that every prediction assumes still air and dry pavement. Courses persist, so re-analysis with a different bike or target costs no re-upload; the original GPX is encrypted on disk exactly as FIT files are, under an opaque storage key rather than an absolute path, and both the rows and the blob are in the data export and go with a delete. The chart profile served alongside the table is ≤400 samples on an **evenly spaced** distance grid, interpolated between recorded points rather than snapped to the nearest one below — a route planner's export is dense through junctions and sparse along the straights, and a chart sizes its marks from the smallest gap in the series
- **Road surface under the course, with honest confidence** — with an optional, self-hosted Valhalla sidecar configured (`VALHALLA_URL`; off by default, tiles built by the self-hoster for their own region), a stored course is map-matched against OSM **in the background** after its segment table has been returned, so upload keeps its latency. Segmentation then splits on surface as well as gradient, and `Crr` comes from the matched class adjusted for tyre width — on asphalt, and wherever the surface is unknown, byte-identical to what the tyre-width curve alone produced, so enabling the sidecar never silently restates an existing plan.

  **Confidence is carried, never flattened**: Valhalla returns `paved_smooth` for a way carrying no surface information at all, so that class reads *inferred*, while every other value requires an explicit `surface`/`tracktype`/`smoothness` tag and reads *confirmed*. An unrecognised value classifies as unknown rather than as a confident-looking default, and overlapping match chunks that disagree downgrade the disputed point. **A short severe sector is not noise**: the dissolve judges on *severity*, not length, so a 40 m snap artefact disappears while 130 m of mud inside 40 km of asphalt keeps its own segment, `Crr`, profile stripe and sentence in the plan. Unconfigured or unreachable, the feature is *absent* rather than broken: the course keeps its Stage 1 analysis and no error reaches the athlete. Stored courses can be enriched with `POST /api/courses/{id}/surface`, and the whole feature sits behind the `allow_course_recon` instance switch, **off by default**, which refuses the capability rather than merely its entry point (the data export excepted, deliberately)
- **The garage — the bikes you own, and what they have done** — `/garage` and `GET /api/bikes` promote the bike row course recon already had (tyre width → Crr, riding position → CdA) into the thing an athlete owns: a starting odometer for the kilometres ridden before openkoutsi saw it, a maintenance log keyed by **component**, the accessories bolted on, and a retirement date for a bike that was sold. Deliberately the same rows the route-analysis bike picker reads, so "bikes in the garage are bikes in the picker" is a fact rather than a synchronisation problem.

  Rides attach themselves: a bike claims a set of cycling `sport_type` values (`default_sports`) and every ingest path — provider sync, file processing, a hand-logged ride, reprocess — runs the same `services/garage.assign_bike`, with `POST /api/bikes/assign-history` for the back catalogue. A sport belongs to **one** bike per athlete; a second claim is a 409 naming the bike that holds it. Assignment is applied rather than suggested (unlike commute detection — no badge counts bikes and nothing is hidden from a prompt), so the safety property lives in `activities.bike_source`: `PATCH /api/activities/{id}` with a `bike_id` stamps `manual`, and no automatic pass will overwrite that, through a reprocess, a fresh sync, a history scan or an edit to what a bike claims. An explicit `null` is a choice too — a rental, a borrowed frame — stamped the same way; `(null, null)` is reserved for "never asked", which is the predicate automapping reads as free to fill.

  Distance is derived on read, never stored, and reported in two figures that are never blurred: `tracked_km` is what openkoutsi observed, `lifetime_km` adds the baseline the athlete typed. Component life falls out of the log — the odometer gap between consecutive entries sharing a component, and `lifetime_km − reading` for the part currently fitted. Retiring a bike keeps its rides, distance and history while dropping it from the pickers; deleting one keeps every ride and course, unassigning rather than destroying them

### AI coaching

- **AI coaching analysis** — per-activity analysis and plan support with OpenAI-compatible backends; cycling activities get a detailed coaching breakdown while other sports are treated as supplemental training and get a short acknowledgement. When an activity is linked to a planned workout, that workout is included in the analysis context so the coach can comment on adherence
- **Koutsi daily feedback** — dashboard card with LLM-generated daily training status covering load trends, recovery state, plan adherence (across all active plans, plus a note for any upcoming plan) and goal progress; auto-triggers after uploads/syncs when enabled
- **AI goal guidance** — on demand, an LLM judges how realistic a goal is for its timeline given current fitness and trend (a `realistic`/`ambitious`/`unrealistic` verdict) and gives concrete coaching steps; streamed in the Koutsi coach voice and persisted per goal
- **Agentic Koutsi** — opt-in (`app_settings.agentic_koutsi`). The daily status card and per-activity analysis stop being a fixed context blob and become an agent loop over the MCP tools: Koutsi decides what it needs and asks, so it can follow a thread — see a flat form number, look at the last four weeks, and say *why*. Because tool rounds emit no prose, the run persists a **progress code** (`thinking`, `tool.get_power_profile`, …) in its own column on the same 500 ms cadence as the text, which the web app localises; a code rather than model prose, so it translates across all fourteen languages and cannot leak tool internals. It is cleared the moment the answer starts.

  Everything degrades to the single-shot prompt at runtime rather than failing — that path is not legacy and stays tested, because under BYOK the hoster does not control whether the athlete's model can call tools. Degradation covers: a provider that rejects the `tools` param, a preset flagged `"tools_supported": false`, a model that accepts `tools` and then calls none, a run that exhausts the round cap (6 for the status card, 3 for one activity) or the 24 000-character tool-result budget, a process already running `AGENT_MAX_CONCURRENT_RUNS` loops, and **any other upstream failure before prose has been written** — a 429, a 5xx, a dropped connection, and above all a context-length 400, which the loop itself creates and small self-hosted windows make likely. The one thing that does *not* degrade is a provider saying our own function schema is invalid: that is our bug, and hiding it behind a quietly worse answer is the failure the rule exists to prevent.

  Bulk provider backlog imports always take the blob path: 4–6× the calls per activity is a real bill on the one path nobody reads one analysis at a time. Tool failures come back as sentences the model can act on, results are truncated with an explicit marker, a turn is held to 4 tool calls (with a sentence for the rest, so the model asks again rather than reasoning from a gap), the instance's `llm_analysis_context` is resent every turn, the `MOOD:` format rule is restated on **every turn that follows tool results** (models obey a leading-format rule less reliably after tool results, and the answering turn is usually the second or third), and token usage is **summed across every call in the run** so the admin usage summary isn't under-reporting. The tools reckon dates from the *athlete's* timezone, so "not due yet" and "missed" agree with the date the prompt asserts
- **Conversational Koutsi** — a dedicated chat page (same `app_settings.agentic_koutsi` opt-in, since a chat that can't look at the athlete's data is a general-purpose LLM in a cycling hat) where the athlete asks their own questions and Koutsi answers by calling the MCP tools. Messages are built **server-side** — the client sends one string, never a message array, which is the whole difference from the `/api/llm/chat` proxy #45 removed, and the reason the scope policy in the system prompt is not something a token holder can edit out.

  That policy is four bands: coaching questions answered fully; adjacent ones (fuelling, sleep, strength, bike fit) answered as a coach rather than refused; medical ones redirected to a clinician without diagnosing or advising training through symptoms; unrelated ones declined in a sentence. It is restated every turn, including after tool results. BYOK caps what any of it can promise: with your own model the output is your model's.

  The prompt carries **the athlete's own clock** — today's date and weekday, the local time and zone, and yesterday's and tomorrow's dates spelled out — because chat is the one surface with no backend-written brief to put it in. It is the same instant the turn's tools reckon from, so the model and its lookups cannot disagree about which day "today" is. Only the dialogue is stored — tool calls and results deliberately are not, since they are most of the bytes, they go stale, and re-running a read-only tool on a later turn is *more* correct than replaying its old answer. What survives is `tool_names`, written through on every progress marker rather than when the turn settles, so the web app can draw each lookup as a step where it happened. Conversations are deletable per-thread and land in the GDPR export as `chat.json`.

  Chat has **no single-shot prompt to fall back on**, so each failure is visible and typed rather than silently degraded: a turn that can't get one of the `AGENT_MAX_CONCURRENT_RUNS` slots *queues* visibly, a model that can't call tools disables the surface up front rather than failing after the athlete has composed a question, and upstream errors settle the turn with a code the web app localises. Per-day and per-conversation turn budgets bound the first LLM surface an athlete can trigger arbitrarily often
- **MCP server** — see [MCP server](#mcp-server) below
- **Inbox** — in-app messages notify athletes about their own events (earned achievements, personal access tokens about to expire or revoked by an admin) and admins about instance events (used invites). Each user has an isolated per-user message store, deletions are permanent, and the design leaves a hook for future email/push delivery. Messages carry their own rendered `title` and `body`, written by `message_text` when the message is sent and stored alongside the `locale` they were rendered in, so a new message type needs no matching template in the web app to be readable

### Integrations and infrastructure

- **Strava + Wahoo sync** — OAuth integrations with history import and webhook updates through bridge services
- **Zone sync** — sync HR/power zones and FTP from connected providers
- **The bridge event queue is bounded** — every accepted webhook is a row kept for seven days, and the main app drains 100 a minute, so a burst that outran the drain grew the SQLite file until the disk did. Both bridges refuse new events past `MAX_QUEUE_EVENTS` (default 10 000, roughly 100 minutes of backlog) with a 503 and a log line. Deliberately refusing rather than evicting: the queued events are the ones about to be processed, so dropping the oldest would lose exactly those, silently
- **Neither bridge starts on a placeholder** — both webhook bridges sit on public HTTPS URLs and used to start happily on the literal default `BRIDGE_SECRET=changeme`, handing the whole event queue to anyone sending `Bearer changeme` (on the Strava bridge that value is also the `hub.verify_token` authorising a subscription). Both now refuse to start on the placeholder or on anything under 32 characters, the same bar `SECRET_KEY` uses, and their bearer checks compare in constant time
- **The unauthenticated router is rate-limited too** — `/api/public` was the only router declaring no limit. Fixing it surfaced a larger problem: slowapi buckets on the *substituted* request path by default, so any limit on a route with a path parameter got a fresh allowance per parameter value and never fired — which silently applied to the admin password-reset mint and both chat write routes as well. The limiter is now keyed per route. Limits key on the address the API *sees*, so `FORWARDED_ALLOW_IPS` and the nginx `limit_req` in `DEPLOY.md` are what make them per client rather than per instance
- **Concurrent-sync safety** — two syncs for one athlete arriving at once (a Wahoo webhook and a Strava backfill milliseconds apart) could interleave in two places, both inside a *single* process. Refreshing an OAuth token is a read-modify-write spanning a network round trip, and Wahoo revokes the old refresh token as it issues the new one, so the loser stored a dead token and the connection stayed broken until reconnected by hand; one caller now claims the rotation with a conditional `UPDATE` and the rest wait and read its result. The ±5-minute duplicate check was guarded only by an `asyncio.Lock` — a statement about one event loop offered in place of one about the database — and is now backed by a lease row (`db/leases.py`) every writer of that database can see, with the lock kept in front as the fast path. Both are reached through a single `activity_create_guard`, so the guarantee is one symbol rather than a shape five call sites must remember. Login also hashes off the event loop, so one sign-in no longer stalls every other request for the duration of a bcrypt
- **Stranded-run recovery** — the three surfaces that write a `pending` LLM status (training status, goal guidance, activity analysis) are judged alive by an **inactivity** clock rather than a start time: the timestamp is touched on every progress commit, so a slow-but-healthy stream is never declared dead underneath itself, and one quiet for 15 minutes is released. Activity analyses carry their own `analysis_updated_at` and gain the age check the other two always had — without it a `pending` row was terminal, since `POST /api/activities/{id}/analyze` refuses while one is in flight. Nothing that sets `pending` survives the process, so an ordinary redeploy strands whatever was mid-run; startup therefore sweeps every per-user database before serving its first request. It settles only the runs whose heartbeat has run down, not every `pending` row: "we just booted" is a claim about the deployment, not this process, and a rolling redeploy overlaps two of them. A run killed by a restart therefore waits out the remainder of its budget, bounded, and is settled by the read that discovers it. Each run also carries a **token** it owns its columns by: settling or re-triggering a row clears the token, so a previous run that is merely slow discards its own writes instead of committing a finished answer over the one that replaced it. The heartbeat says whether a run is *alive*; the token says whether its writes are still *wanted*
- **Swappable email module** — a provider-agnostic seam (`backend/app/services/email/`) for all email: a generic `EmailProvider` interface (outbound `send`, inbound `verify_inbound_signature`/`parse_inbound`) with `LettermintProvider` and `EuromailProvider` (euromail.dev — EU-based, inbound on its free tier) implementations, provider selection via `EMAIL_PROVIDER`, and self-rendered inline-styled HTML + plain-text bodies. Optional — with no provider configured, email-dependent features stay unavailable rather than erroring
- **Cycling-themed 404 page** — localized "Wrong Turn!" not-found page with cycling flavour

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
| Reading an activity file, whatever its format | `openkoutsi/activity_formats.py` (registry) + `gpx.py` / `tcx.py` / `fit.py` |
| Unpacking a zip or gzip from a stranger, safely | `services/activity_archive.py` |
| Working through a bulk import and reporting on each file | `services/activity_import.py` |
| Handing a webhook event to exactly one consumer, without losing it | `services/bridge_client.py` + each bridge's `/events/claim` — a claim with a deadline, acked on success and nacked on failure, so a consumer that dies mid-import gets the event redelivered |
| Electing which process runs the background pollers | `services/leadership.py` — one claim (`background-work`) on a registry lease, taken per cycle rather than as a term of office, cancelling the cycle in flight if it is lost |
| Serialising a write section across processes, not just across tasks | `db/leases.py` (`hold`), reached through `provider_sync.activity_create_guard` — taken by all five activity writers: provider sync, single upload, bulk import and both webhook paths |
| Bringing a user's database into existence | `db/user_session.py` (`init_user_db`) — the only place that creates one; it stamps the new file at the current Alembic head so the next deploy skips it. Getting an engine is side-effect-free, so no read path can conjure a directory from an id it was handed |
| What an AI coach may ask for, and what it gets back | `mcp/registry.py` (declarations) + `mcp/dispatch.py` (every check) |

## Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12 · FastAPI · SQLAlchemy 2 (async) · Alembic |
| Database | SQLite (WAL mode) |
| Auth | JWT (`python-jose` · `bcrypt`) |
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
# Required — the backend will not start without it (or ALLOW_PLAINTEXT_SECRETS=true):
ENCRYPTION_KEY=<fernet-key>
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

### Tests

```bash
uv run python -m pytest tests/
```

Every test gets its own registry and per-user database, built fresh in memory. The
`CREATE` statements for both schemas are compiled once per session and replayed per
test rather than re-derived from the ORM metadata each time (`create_all` spends
~65 ms per test compiling DDL that SQLite executes in under 2 ms). They are read back
out of `sqlite_master` after a real `create_all`, so they are exactly what SQLAlchemy
emits and there is no second description of the schema to drift.

CI measures coverage with `COVERAGE_CORE=sysmon` (`sys.monitoring`, Python 3.12+)
rather than the default C trace function, and uploads `coverage.xml` to Codecov — the
source of the badge above. The tracer is both slower (6m22 against 4m04) and wrong
here, reporting **80%** where `sys.monitoring` reports **91%**. The gap is almost
entirely `backend/app/api/*` — `auth.py` 44% against 92%, `chat.py` 52% against 98%,
`courses.py` 47% against 95% — because those are `async` route handlers whose database
work runs inside greenlets via SQLAlchemy's asyncio layer, and a stack-based trace
function loses frames across a greenlet switch. 1361 lines across 40 files were being
under-counted that way.

## Environment variables

Main app (`.env`):

```env
# Required
SECRET_KEY=<random 256-bit key>

# Core settings
DATA_DIR=data
FRONTEND_URL=http://localhost:3000
API_URL=http://localhost:8000

# Required. Encrypts stored OAuth tokens, FIT files and LLM API keys. Set
# ALLOW_PLAINTEXT_SECRETS=true instead only for a throwaway instance — it
# stores provider tokens unencrypted and warns on every start.
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

# Road surface classification (optional, issue #56) — a Valhalla sidecar you run
# yourself, reachable only from inside the deployment. Unset by default; with it
# unset a course is solved as dry pavement, which the written plan says out loud.
# Building tiles takes hours and gigabytes; see the openkoutsi-ops repository.
# Course recon itself is gated by `allow_course_recon`, which defaults **off**.
VALHALLA_URL=                        # e.g. http://valhalla:8002

# Email (optional) — outbound mail and inbound webhooks go through the swappable
# email module (backend/app/services/email/). With no provider configured,
# email-dependent features stay unavailable.
EMAIL_PROVIDER=lettermint            # provider selection; "lettermint" or "euromail"
EMAIL_FROM=                          # sender address for outbound mail
LETTERMINT_API_KEY=                  # Lettermint API token for sending
LETTERMINT_WEBHOOK_SECRET=           # secret for verifying inbound Lettermint webhooks
EUROMAIL_API_KEY=                    # EuroMail API token for sending (EMAIL_PROVIDER=euromail)
EUROMAIL_WEBHOOK_SECRET=             # secret for verifying inbound EuroMail webhooks

# Optional: restrict which LLM base URLs users may bring (BYOK). Comma-separated;
# empty = users may bring any URL (subject to SSRF guards).
LLM_ALLOWED_SERVERS=

# Set to true if your LLM runs on localhost (Ollama) or the LAN. The SSRF guard
# refuses base URLs resolving into loopback/private/CGNAT ranges by default;
# cloud metadata ranges stay blocked regardless.
LLM_ALLOW_PRIVATE_NETWORKS=false

# Optional: path to the dedicated LLM-usage database (per-call token accounting
# for instance-paid calls). Empty = <DATA_DIR>/llm_usage.db.
LLM_USAGE_DB=
```

There are no server-side LLM env-var defaults: all LLM connections are defined as
presets — instance-wide by an admin, or per-user via BYOK (see below).

The backend deliberately sends no `temperature` parameter, so each model applies its
own default. That keeps thinking-enabled models (e.g. Claude with extended thinking,
via Anthropic's OpenAI-compatible endpoint), which reject any temperature other than
`1`, working out of the box. Upstream LLM errors surface the provider's response body
in the logs.

**Presets.** Admins configure, per instance (Settings → AI / LLM), a list of
selectable presets — each a self-contained connection: display name, stable
identifier, base URL, model id, API key, headers and extra chat-completion body
params (e.g. `max_tokens` or a `reasoning_effort` config), so an admin can offer
distinct providers side by side. **The first preset in the list is the instance
default.** The dropdown shows display names but stores the stable identifier, so
renaming one never breaks existing selections. A user's selected preset (or BYOK
server) is honoured everywhere an LLM is called on their behalf — the plan/workout
generators **and** the background analysers — falling back to the instance default
only when they haven't chosen one.

**Bring your own LLM (BYOK).** Any user can instead point openkoutsi at their own
OpenAI-compatible endpoint under Settings → AI / LLM (base URL + model + optional API
key). Once a user sets their own base URL, **only** their config is used — the
instance's presets and keys are ignored entirely, so an instance key can never be sent
to a user-chosen server. The API key is Fernet-encrypted per-user at rest and never
returned to the browser. When `LLM_ALLOWED_SERVERS` is set, BYOK URLs are restricted to
that allow-list at save time and at use time; the SSRF guard always applies.

**Guaranteed JSON for structured generation.** Plan and structured-workout generation
send a strict JSON-schema `response_format` derived once from the backend's own
pydantic models, so models supporting structured outputs are constrained to the exact
shape the parsers accept. On by default; a provider that rejects the parameter is
detected and the call transparently retried without it, and a preset can pre-empt that
with `structured_outputs: false`. Prompt instructions plus parse-and-retry remain the
final safety net.

**Tool calling for the agentic coach** has the same three-part shape, for the same
reason: under BYOK the hoster does not control whether the athlete's server can do
function calling. `tools` is sent by default; a provider that rejects the *parameter*
is detected at runtime and the run falls back to the single-shot blob prompt; and a
preset can pre-empt that with `tools_supported: false`, which is the flag that matters
for a server that accepts `tools` and then emits nonsense, since that one never
produces the 400 runtime detection needs. Neither detector swallows an "invalid schema"
body: that means *our* pydantic model is broken.

**SSRF guard.** It resolves the hostname, refuses the request if *any* returned address
is in a blocked range, and connects to the address it vetted rather than re-resolving
the name (which a short-TTL record could answer differently). Cloud metadata,
link-local and multicast ranges are always refused; loopback and private ranges too
unless `LLM_ALLOW_PRIVATE_NETWORKS=true`, which is the switch for a model self-hosted
on localhost or the LAN. A failing upstream's response body is echoed back only on the
admin test, since the BYOK test's URL comes from the caller.

The connection tests — *Test connection* (admin, instance presets) and the user BYOK
card's (`POST /api/llm/test-my-connection`) — send a small "hello world" using the
configured headers and the selected model's body params, so they validate ZDR headers
and a thinking config, not just reachability.

**LLM subscription gating + usage tracking (opt-in).** An admin can flip
`llm_requires_subscription` (Settings → AI / LLM) to require an "LLM access"
entitlement for the *instance's* LLM credentials. It defaults **off**. When on, users
without an entitlement can still use every LLM feature via BYOK, or receive a
machine-readable `llm_subscription_required` 403 the frontend turns into an upsell.
Admins grant/revoke entitlements per user (`PUT /api/admin/users/{id}/llm-entitlement`);
`GET /api/llm/access` is the frontend's source of truth. Independently, every
**instance-paid** call's token usage (input and output counted separately, plus
provider and model) is recorded in a separate database (`LLM_USAGE_DB`, default
`data/llm_usage.db`), so the hoster can compute average cost per user over any period
via `GET /api/admin/llm-usage/summary` (day/week/month buckets). BYOK calls are never
recorded — the user pays their own provider.

The web frontend has its own configuration (`API_URL`, etc.) — see the [openkoutsi-web](https://github.com/openkoutsi/openkoutsi-web) repository.

## Integrations

- **Strava:** configure Strava app credentials in `.env` and deploy `strava_bridge/` to a public HTTPS URL. The bridge's `POST /webhook` is **unauthenticated**: Strava documents no webhook signing, so the `X-Hub-Signature-256` check is off by default — requiring a header Strava never sends refused every real event with `401`. In its place: activity-only filtering, the unknown-owner drop in the main app, re-fetching each activity from Strava's API, and the `MAX_QUEUE_EVENTS` ceiling. The check stays behind `STRAVA_VERIFY_WEBHOOK_SIGNATURE=true` for the day Strava documents a validation sequence.
- **Wahoo:** configure Wahoo credentials in `.env` and deploy `wahoo_bridge/` to a public HTTPS URL. Pushing structured workouts to Wahoo requires the `plans_read`, `plans_write`, and `workouts_write` scopes; users connected before this feature must reconnect Wahoo to grant them. The "Generate workouts" plan action needs a server-reachable LLM (resolved athlete → instance → global) to synthesize the structured workouts; uploading the generated workouts to Wahoo is then done individually from the Workouts tab.
- **Disconnecting a provider:** `DELETE /api/integrations/{provider}/disconnect` also deletes the imported activities when `delete_data=true` is passed (as a query parameter *or* in the JSON body). The data is deleted and committed *before* the connection is removed, and a failed deletion returns `500` with the connection left in place — the caller is never told the data is gone unless it was.

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

`POST /mcp` speaks the Model Context Protocol (revision `2025-06-18`) over JSON-RPC
2.0, stateless: `initialize`, `notifications/initialized`, `ping`, `tools/list` and
`tools/call`. Tools only — resources and prompts are absent from the advertised
capabilities rather than stubbed, so a client never offers a user something that will
fail.

Ten read-only, task-shaped coaching tools publish one athlete's own data:
`get_training_status`, `find_activity`, `list_recent_activities`,
`get_activity_detail`, `get_plan_status`, `get_goal_progress`, `get_power_profile`,
`get_intensity_distribution`, `get_zone_totals` and `get_athlete_profile`. Instead of
a prompt builder guessing what a coach needs and assembling a fixed blob, the model
asks.

`get_training_status` answers for a **past date** as well as today (`as_of`), so "what
shape was I in before that event" and "is this build steeper than the last one" are
two comparable calls. Every figure moves with the date, the trailing volume window
included, and a date before the athlete's recorded history is refused with the date
that history starts on rather than answered with zeros a model would read as a
collapse.

`get_athlete_profile` is the one tool about the athlete rather than about training
that happened: the power and heart-rate **zone boundaries** every zone figure from the
other tools is measured against, plus the physiology, the hours a week they say they
have, and the coaching tone they asked for. It is deliberately not a profile dump — no
name, no date of birth (an age in whole years instead), no FTP-test or weight history,
no provider connections, no feature toggles and none of the BYOK model configuration.
It is also the only tool `athlete:read` opens on its own.

Tools return **computed aggregates, never raw streams** (a three-hour ride holds
~11 000 samples per stream), preserve reason codes rather than flattening them to
nulls (`too_short`, `no_power`, …), name the unit in every field description, bound
every collection and report the true `total` alongside it, and never return
coordinates. Failures come back as readable tool *results*, not exceptions — "No
activity on 2026-07-14. Nearest rides: 2026-07-13 (endurance, 2 h 04) and 2026-07-16
(threshold, 1 h 12)."

Two consumers reach the identical tools through different doors — the on-server agent,
and external MCP clients authenticating with a personal access token — and both pass
the same checks. Because the endpoint resolves its own credential (no single scope
could be honest about ten differently-scoped tools), the route-policy walk never covers
it, so the registry carries **its own default-deny**: a tool that declares no scopes
cannot be registered, and a test proves it. Consent is checked per invocation, calls
are rate-limited per user (the in-process agent deliberately is not), and every
invocation is audited with caller, tool, arguments, duration and outcome — never the
result.

**Genuinely read-only**: the two zone tools are explicitly asked not to freeze missing
time-in-zone snapshots the way their REST counterparts do, and report the rides they
couldn't count instead. A snapshot is permanent, so letting a `metrics:read` tool
trigger one would let the moment an agent asked a question decide forever which zone
definitions an old ride is judged against, and would make the `readOnlyHint` an MCP
client uses to decide whether to prompt the user a lie.

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
regardless — hiding the unreachable ones would make a scope refusal look like a missing
feature — with the scopes each needs in its `_meta`, so a client can explain the gap
rather than discover it by failing. There is **no `mcp:*` scope**, which would obscure
the actual grant: all ten tools are covered by the five read scopes
`activities:read`, `athlete:read`, `goals:read`, `metrics:read` and `plans:read`, and
each of those five opens at least one tool on its own. `athlete:export` is deliberately
**not** callable — one call returning the whole record is the opposite of task-shaped.

An admin can turn the endpoint off instance-wide with `allow_mcp_server` (the **Allow
the MCP server** switch in the admin console's Settings tab, default **on**), which
refuses it outright, handshake included, rather than letting a client connect to a
server that will decline every useful call. Denying `/mcp` at the reverse proxy works
too. Either way it narrows the *interface*, not the exposure: the same data is
reachable through the ordinary REST routes with the same token, and what limits a
credential is its scopes. See [DEPLOY.md](DEPLOY.md).

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
