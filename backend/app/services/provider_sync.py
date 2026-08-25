"""
Generic provider sync pipeline.

Works with any provider registered in the PROVIDERS registry. The logic is
identical regardless of source: refresh tokens, paginate activities, find or
create the single Activity record for this real-world workout, attach an
ActivitySource row, and (re)populate the Activity's metrics if the new source
has higher priority than whatever was there before.

Priority (lower = higher priority):
  1  upload   — manual FIT upload
  2  wahoo    — Wahoo cloud sync with a FIT file
  3  strava   — Strava API (stream-based)
  4  wahoo    — Wahoo cloud sync without a FIT file (blank)
  5  manual   — manually entered activity
"""

import asyncio
import dataclasses
import io
import logging
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncIterator

from sqlalchemy import delete, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.file_encryption import encrypt_file
from backend.app.db import leases
from backend.app.models.registry_orm import ProviderConnection
from backend.app.models.user_orm import (
    Activity,
    ActivityDistanceBest,
    ActivityInterval,
    ActivityPowerBest,
    ActivitySource,
    ActivityStream,
    Athlete,
    SyncLease,
)
from backend.app.services.stranded_runs import begin_activity_analysis_run
from openkoutsi.categorization import classify_workout
from openkoutsi.fit_processing import (
    resolve_sport_type,
    auto_interval_s,
    build_auto_intervals,
    compute_interval_stats,
)
from backend.app.services.providers.base import NormalizedActivity
from backend.app.services.providers.registry import PROVIDERS
from backend.app.services.weight import effective_weight_for, load_weight_log, w_per_kg
from backend.app.services.aerobic_metrics import apply_aerobic_metrics
from openkoutsi.training_math import (
    calculate_load,
    compute_distance_bests,
    compute_power_bests,
    compute_torque_stream,
    variability_index,
    weighted_power,
)
from openkoutsi.activity_formats import parser_for
from openkoutsi.fit import summarizeWorkout
from openkoutsi.streams import to_json_stream

log = logging.getLogger(__name__)

_DUPLICATE_WINDOW = timedelta(minutes=5)

# Sentinel: _fill_from_source uses this to know FIT hasn't been fetched yet
_NOTFETCHED = object()

# How long the activity-creation lease may be held before another caller may
# take it over. Generous because the section it covers can include a FIT
# download and parse on the attach path — expiring under a holder that is merely
# slow would hand the same lease to two callers, which is the duplicate this
# exists to prevent. See `backend.app.db.leases`.
_ACTIVITY_LEASE_TTL = timedelta(minutes=5)
_ACTIVITY_LEASE_WAIT = 60.0

# Per-(user_id, athlete_id) lock that serialises the dedup-window-query +
# create/attach operation. Prevents the race condition where two concurrent
# syncs both see "no existing activity" and each create a new one for the
# same real-world workout.
#
# This is the *in-process* half of that guard; the durable half is the
# `SyncLease` taken inside it (issue #50). Keeping both is deliberate: this one
# is free and settles the common case without touching the database, and the
# lease is what makes the guarantee true beyond one event loop.
#
# Bounded, because it is keyed by athlete and never had an eviction path: on a
# long-lived process syncing many users it only grew. Entries are dropped oldest
# first and *only while unheld* — evicting a held lock would hand the same
# athlete two different locks and silently undo the mutual exclusion.
_MAX_ACTIVITY_LOCKS = 256

# The loop is cached alongside each lock because an `asyncio.Lock` is meaningful
# only to the loop it was created on — using one from another loop raises. A
# server process has exactly one loop for its lifetime, so this never fires in
# production; it is what keeps a *cache* from outliving the thing it caches.
_activity_creation_locks: OrderedDict[
    tuple[str, str], tuple[asyncio.AbstractEventLoop, asyncio.Lock]
] = OrderedDict()


def _get_activity_lock(user_id: str, athlete_id: str) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    key = (user_id, athlete_id)

    cached = _activity_creation_locks.get(key)
    if cached is not None and cached[0] is loop:
        _activity_creation_locks.move_to_end(key)
        return cached[1]

    def _evictable(entry: tuple[asyncio.AbstractEventLoop, asyncio.Lock]) -> bool:
        # A lock from another loop cannot be held by anything that is running.
        return entry[0] is not loop or not entry[1].locked()

    while len(_activity_creation_locks) >= _MAX_ACTIVITY_LOCKS:
        stale = next(
            (k for k, v in _activity_creation_locks.items() if _evictable(v)), None
        )
        if stale is None:
            break  # every lock is in use; the cap yields rather than corrupt.
        del _activity_creation_locks[stale]

    lock = asyncio.Lock()
    _activity_creation_locks[key] = (loop, lock)
    return lock


# ── The activity-create guard ────────────────────────────────────────────────


@asynccontextmanager
async def activity_create_guard(
    session: AsyncSession, user_id: str, athlete_id: str
) -> AsyncIterator[None]:
    """Serialise one athlete's dedup-window query against the create that follows.

    Two providers can deliver the same ride almost simultaneously — a Wahoo
    webhook and a Strava sync firing within milliseconds — and without this both
    callers see an empty ±5-minute window and each create the same activity.

    **Two guards, not one** (issue #50). The ``asyncio.Lock`` settles the common
    case without touching the database, but it only speaks for this event loop;
    the ``SyncLease`` inside it repeats the same exclusion in a place every
    writer of this database can see. The lock is the fast path, the lease is the
    guarantee — defence in depth rather than either one alone.

    **The invariant every caller owes this guard:** the new ``Activity`` row must
    be *committed* before the block exits. A flush is not sufficient — under
    SQLite's WAL isolation a caller that takes the lease next still sees an empty
    window until the write is committed, which is the duplicate this exists to
    prevent.

    ``leases.hold`` owns the session's transaction boundaries for the duration,
    so do not carry unrelated uncommitted work across this block. In exchange, a
    block that raises is rolled back before the lease is released — which is what
    keeps a failed attach path from publishing its own deletions.

    Known exposure, unchanged by this refactor: the Wahoo and provider-sync
    attach paths download a FIT inside the block, so a pathologically slow CDN
    can outlast ``_ACTIVITY_LEASE_TTL`` and hand the same lease to a second
    caller. Hoisting the prefetch out would reorder the priority decision that
    depends on it, so it is left as a follow-up rather than fixed here.
    """
    async with (
        _get_activity_lock(user_id, athlete_id),
        leases.hold(
            session,
            SyncLease,
            f"activity-create:{athlete_id}",
            ttl=_ACTIVITY_LEASE_TTL,
            wait=_ACTIVITY_LEASE_WAIT,
        ),
    ):
        yield


# ── Priority ──────────────────────────────────────────────────────────────────


def _source_priority(provider: str, has_fit: bool) -> int:
    """Lower number = higher priority."""
    if provider == "upload":
        return 1
    if provider == "wahoo" and has_fit:
        return 2
    if provider == "strava":
        return 3
    if provider == "wahoo":  # no FIT file
        return 4
    return 5  # manual, unknown


def _winning_priority(activity: Activity) -> int:
    """Priority of the source currently populating this Activity's metrics."""
    if not activity.sources:
        return 999
    return min(
        _source_priority(s.provider, bool(s.fit_file_path)) for s in activity.sources
    )


# ── Token management ──────────────────────────────────────────────────────────

# How far ahead to refresh before actual expiry, per provider.
# Strava tokens last 6 h — refresh when ≤30 min remain (Strava's own recommendation).
# Wahoo tokens last 2 h — 1 min is enough; Wahoo revokes old tokens on refresh so
# we refresh as late as possible to avoid unnecessary rotations.
_REFRESH_LOOKAHEAD: dict[str, timedelta] = {
    "strava": timedelta(minutes=30),
    "wahoo": timedelta(minutes=1),
}
_DEFAULT_REFRESH_LOOKAHEAD = timedelta(minutes=1)

# How long a claimed rotation may run before another caller may take it over.
# Generously longer than a refresh round trip: the cost of it being too short is
# a double rotation, which is the bug this exists to prevent. It is a
# crash-recovery bound, not a timeout — the happy path always releases early.
_REFRESH_LOCK_TTL = timedelta(seconds=60)
# How long a caller that lost the claim waits for the winner's tokens before
# giving up and using whatever is on the row.
_REFRESH_WAIT_SECONDS = 30.0
_REFRESH_POLL_SECONDS = 0.05
# Claim-or-wait turns before a caller settles for the token on the row. Two: one
# to wait for the caller already rotating, one to rotate itself if that failed.
_REFRESH_ATTEMPTS = 2


def _needs_refresh(conn: ProviderConnection, now: datetime | None = None) -> bool:
    """Whether this connection's access token is close enough to expiry to rotate."""
    expires_at = conn.token_expires_at
    if expires_at is None or not conn.refresh_token:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    lookahead = _REFRESH_LOOKAHEAD.get(conn.provider, _DEFAULT_REFRESH_LOOKAHEAD)
    return (now or datetime.now(timezone.utc)) + lookahead >= expires_at


async def _claim_refresh(conn: ProviderConnection, session: AsyncSession) -> bool:
    """Try to become the one caller that rotates this connection's tokens.

    A conditional UPDATE, so the *database* picks the winner. That is what makes
    this hold between processes as well as between tasks — and the rotation is
    already worth serialising within one process, since two syncs for the same
    user interleave across the ``await`` on the provider's refresh endpoint.
    """
    now = datetime.now(timezone.utc)
    result = await session.execute(
        update(ProviderConnection)
        .where(
            ProviderConnection.id == conn.id,
            or_(
                ProviderConnection.refresh_lock_until.is_(None),
                ProviderConnection.refresh_lock_until <= now,
            ),
        )
        .values(refresh_lock_until=now + _REFRESH_LOCK_TTL)
        # The database decides the winner; nothing in the identity map should be
        # second-guessing that, and SQLite hands these columns back naive, which
        # the in-Python evaluator cannot compare against an aware `now` anyway.
        .execution_options(synchronize_session=False)
    )
    await session.commit()
    return result.rowcount == 1


async def _release_refresh(conn_id: str, session: AsyncSession) -> None:
    """Hand the claim back. Takes the id, not the row: the caller may have just
    rolled back, which expires every instance in the session."""
    await session.execute(
        update(ProviderConnection)
        .where(ProviderConnection.id == conn_id)
        .values(refresh_lock_until=None)
        .execution_options(synchronize_session=False)
    )
    await session.commit()


async def _await_rotation(conn: ProviderConnection, session: AsyncSession) -> None:
    """Wait for whoever won the claim, so ``conn`` ends up holding *their* tokens.

    The rollback on each pass is load-bearing: registry sessions are
    ``expire_on_commit=False`` and SQLite hands out a snapshot for the life of a
    read transaction, so without ending the transaction first this would re-read
    its own stale view forever and hand back the token the winner has just had
    revoked.
    """
    deadline = asyncio.get_running_loop().time() + _REFRESH_WAIT_SECONDS
    while True:
        await asyncio.sleep(_REFRESH_POLL_SECONDS)
        await session.rollback()
        await session.refresh(conn)
        lock_until = conn.refresh_lock_until
        if lock_until is not None and lock_until.tzinfo is None:
            lock_until = lock_until.replace(tzinfo=timezone.utc)
        if lock_until is None or lock_until <= datetime.now(timezone.utc):
            return
        if asyncio.get_running_loop().time() >= deadline:
            log.warning(
                "Timed out waiting for a concurrent %s token refresh for user %s",
                conn.provider,
                conn.user_id,
            )
            return


async def ensure_fresh_token(conn: ProviderConnection, session: AsyncSession) -> str:
    """Refresh the access token if it will expire soon. Returns current token.

    At most one caller rotates at a time. The others wait for it and return the
    token it stored, rather than presenting a refresh token the provider has
    already revoked — Wahoo revokes on rotation (see ``_REFRESH_LOOKAHEAD``
    above), so a lost race used to cost the user their connection permanently
    (issue #50).

    Waiting is not the same as giving up. If the winner's attempt fails it
    releases the claim, and a caller that was already waiting takes its own turn
    rather than returning a token it can see is expiring — otherwise one
    transient 5xx would take down every caller queued behind it, which is worse
    than the independent attempts this replaced.

    The one path that still returns a token known to be stale is
    ``_await_rotation`` timing out; it logs, and the caller then fails at the
    provider exactly as it did before this existed.
    """
    if not _needs_refresh(conn):
        return conn.access_token or ""

    client_cls = PROVIDERS.get(conn.provider)
    if client_cls is None:
        log.warning("Unknown provider %s — cannot refresh token", conn.provider)
        return conn.access_token or ""

    # One turn to wait for whoever is already rotating, one to do it ourselves if
    # their attempt failed. Bounded, so two callers failing in turn cannot
    # ping-pong indefinitely between waiting and retrying.
    for _ in range(_REFRESH_ATTEMPTS):
        if await _claim_refresh(conn, session):
            return await _rotate_under_claim(conn, session, client_cls)
        await _await_rotation(conn, session)
        if not _needs_refresh(conn):
            return conn.access_token or ""
    return conn.access_token or ""


async def _rotate_under_claim(
    conn: ProviderConnection, session: AsyncSession, client_cls
) -> str:
    """Do the rotation, holding the claim. Returns the resulting access token."""
    # Read off the row before anything can expire it — a rollback below detaches
    # every loaded attribute, and reloading one would need IO we may not have.
    conn_id, user_id = conn.id, conn.user_id
    try:
        # Re-read under the claim. Between deciding a refresh was due and winning
        # the right to do it, another process may have rotated already — and
        # rotating a second time is exactly what this lock exists to prevent.
        await session.refresh(conn)
        if not _needs_refresh(conn):
            token = conn.access_token or ""
            await _release_refresh(conn_id, session)
            return token

        try:
            tokens = await client_cls.refresh_access_token(conn.refresh_token)  # type: ignore[arg-type]
        except Exception:
            log.error(
                "Failed to refresh %s token for user %s",
                conn.provider,
                conn.user_id,
                exc_info=True,
            )
            raise

        conn.access_token = tokens["access_token"]
        conn.refresh_token = tokens["refresh_token"]
        conn.token_expires_at = datetime.fromtimestamp(
            tokens["expires_at"], tz=timezone.utc
        )
        # Released in the same commit that publishes the new tokens, so no waiter
        # can ever observe a free lock next to the tokens it replaced.
        conn.refresh_lock_until = None
        await session.commit()
        log.info("Refreshed %s token for user %s", conn.provider, user_id)
    except Exception:
        # A failed rotation must not hold the claim for the whole TTL: the next
        # caller should get to try, not wait a minute to find out it may.
        # Cancellation and process death are deliberately *not* handled here —
        # releasing needs IO that a cancelled task cannot do, which is precisely
        # what the deadline on the claim is for.
        try:
            await session.rollback()
            await _release_refresh(conn_id, session)
        except Exception:
            log.exception("Could not release the refresh lock for user %s", user_id)
        raise

    return conn.access_token or ""


# ── Full sync ─────────────────────────────────────────────────────────────────


async def sync_provider_activities(
    athlete: Athlete,
    connection: ProviderConnection,
    session: AsyncSession,
    *,
    user_id: str,
    access_token: str,
) -> tuple[int, date | None]:
    """
    Import all activities from a provider that aren't already in the database.

    For each activity from the provider:
      - If this (provider, external_id) pair already has an ActivitySource → skip.
      - If an Activity exists within ±5 min → attach a new ActivitySource and
        repopulate the Activity if the new source has higher priority.
      - Otherwise → create a new Activity + ActivitySource.

    ``access_token`` is the caller's to supply, and required. ``session`` here is
    the **per-user** database — every model this function touches belongs to it —
    while ``connection`` is a registry row. Refreshing a token therefore needs a
    session this function does not have, so it used to accept ``None`` and call
    :func:`ensure_fresh_token` with the wrong session: harmless while that only
    mutated the ORM object, a ``no such table: provider_connections`` now that it
    claims the rotation with a real statement. Every caller already passes a
    token; the signature now says so.

    Returns (count_created_or_updated, earliest_start_date).
    """
    provider_name = connection.provider
    client_cls = PROVIDERS.get(provider_name)
    if client_cls is None:
        log.error("No client registered for provider %s", provider_name)
        return 0, None

    client = client_cls()

    count = 0
    earliest: date | None = None
    page = 1

    while True:
        activities = await client.list_activities(access_token, page)
        if not activities:
            break

        for norm in activities:
            ext_id = norm.external_id

            # ── Already imported this (provider, external_id)? ────────────
            src_result = await session.execute(
                select(ActivitySource)
                .join(Activity, ActivitySource.activity_id == Activity.id)
                .where(
                    Activity.athlete_id == athlete.id,
                    ActivitySource.provider == provider_name,
                    ActivitySource.external_id == ext_id,
                )
            )
            existing_src = src_result.scalar_one_or_none()
            if existing_src is not None:
                # Handle duration correction (moving_time preference)
                act = existing_src.activity
                if (
                    norm.duration_s
                    and act.duration_s
                    and norm.duration_s < act.duration_s
                ):
                    old_dur = act.duration_s
                    act.duration_s = norm.duration_s
                    if act.weighted_power and athlete.ftp:
                        new_tss, new_if = calculate_load(
                            norm.duration_s,
                            act.weighted_power,
                            act.avg_hr,
                            athlete.ftp,
                            athlete.max_hr,
                        )
                        act.load = new_tss
                        act.intensity = new_if
                    elif act.avg_hr and athlete.max_hr:
                        new_tss, _ = calculate_load(
                            norm.duration_s,
                            None,
                            act.avg_hr,
                            athlete.ftp,
                            athlete.max_hr,
                        )
                        act.load = new_tss
                    await session.commit()
                    log.info(
                        "Corrected duration for %s/%s: %ds → %ds",
                        provider_name,
                        ext_id,
                        old_dur,
                        norm.duration_s,
                    )
                continue

            # ── Find-or-create under the activity-create guard ────────────
            # The attach branch below downloads a FIT inside the block, which is
            # what `_ACTIVITY_LEASE_TTL` is sized for.
            async with activity_create_guard(session, user_id, athlete.id):
                # ── Activity within the time window? ──────────────────────
                if norm.start_time is not None:
                    act_result = await session.execute(
                        select(Activity).where(
                            Activity.athlete_id == athlete.id,
                            Activity.start_time >= norm.start_time - _DUPLICATE_WINDOW,
                            Activity.start_time <= norm.start_time + _DUPLICATE_WINDOW,
                        )
                    )
                    existing_act = act_result.scalar_one_or_none()
                else:
                    existing_act = None

                if existing_act is not None:
                    # Guard: if the existing activity already has a source from
                    # this same provider (but a different external_id), these are
                    # two distinct workouts that both fall inside the dedup window
                    # (e.g. a warm-up and a main ride starting 3 min apart, both
                    # on Strava). The (activity_id, provider) unique constraint
                    # would fire if we tried to attach a second source from the
                    # same provider to the same activity. Treat the incoming
                    # activity as a separate workout by clearing existing_act and
                    # falling through to the "new workout" path below.
                    if any(s.provider == provider_name for s in existing_act.sources):
                        existing_act = None

                if existing_act is not None:
                    # Same real-world workout from a different provider — attach a new source.
                    new_src = ActivitySource(
                        activity_id=existing_act.id,
                        provider=provider_name,
                        external_id=ext_id,
                    )
                    session.add(new_src)
                    await session.flush()

                    # Pre-fetch FIT to determine actual priority before deciding
                    # whether to repopulate. This avoids the bug where Wahoo with
                    # FIT (priority=2) would be skipped because the pessimistic
                    # priority (no FIT, priority=4) doesn't beat Strava (priority=3).
                    prefetched_fit: bytes | None = None
                    try:
                        prefetched_fit = await client.download_fit_file(
                            access_token, norm.external_id
                        )
                    except Exception:
                        prefetched_fit = None

                    actual_priority = _source_priority(
                        provider_name, prefetched_fit is not None
                    )
                    if actual_priority < _winning_priority(existing_act):
                        await _repopulate_activity(
                            existing_act,
                            new_src,
                            norm,
                            client,
                            access_token,
                            athlete,
                            session,
                            user_id=user_id,
                            prefetched_fit=prefetched_fit,
                        )
                        count += 1
                        if existing_act.start_time:
                            day = (
                                existing_act.start_time.date()
                                if hasattr(existing_act.start_time, "date")
                                else existing_act.start_time
                            )
                            if earliest is None or day < earliest:
                                earliest = day
                    else:
                        # Lower priority — just record the source, don't touch metrics.
                        await session.commit()
                    continue

                # ── New workout — create Activity + ActivitySource ─────────
                activity = Activity(
                    athlete_id=athlete.id,
                    name=norm.name,
                    sport_type=norm.sport_type,
                    start_time=norm.start_time,
                    duration_s=norm.duration_s,
                    distance_m=norm.distance_m,
                    elevation_m=norm.elevation_m,
                    avg_power=norm.avg_power,
                    avg_hr=norm.avg_hr,
                    max_hr=norm.max_hr,
                    avg_speed_ms=norm.avg_speed_ms,
                    avg_cadence=norm.avg_cadence,
                    status="pending",
                )
                session.add(activity)
                await session.flush()

                src = ActivitySource(
                    activity_id=activity.id,
                    provider=provider_name,
                    external_id=ext_id,
                )
                session.add(src)
                await session.flush()

                # Commit inside the lock so the Activity is visible to any
                # concurrent session that next acquires the lock and queries
                # the dedup window.  _populate_activity will update the row
                # again (metrics, streams, status) and commit a second time.
                await session.commit()

            # FIT download and stream processing happen outside the lock —
            # they are slow I/O operations that don't need to be serialised.
            await _populate_activity(
                activity, src, norm, client, access_token, athlete, session, user_id=user_id
            )
            count += 1

            if activity.start_time:
                day = (
                    activity.start_time.date()
                    if hasattr(activity.start_time, "date")
                    else activity.start_time
                )
                if earliest is None or day < earliest:
                    earliest = day

            app_cfg = athlete.app_settings or {}
            if app_cfg.get("auto_analyze"):
                from backend.app.services.llm_access import auto_analysis_allowed
                from backend.app.services.llm_activity_analyzer import (
                    analyze_activity_bg,
                )

                # Issue #9: skip the instance-paid auto analysis for denied users
                # on a gated instance.
                if await auto_analysis_allowed(user_id, athlete):
                    run_id = begin_activity_analysis_run(activity)
                    await session.commit()
                    # Issue #43: a backlog import is the one path where an agent loop's
                    # 4–6× calls is a real bill and nobody reads the output one by
                    # one, so it always takes the single-shot prompt.
                    asyncio.create_task(
                        analyze_activity_bg(
                            activity.id, athlete.id, user_id,
                            allow_agentic=False, run_id=run_id,
                        )
                    )

        page += 1

    return count, earliest


# ── Data population ───────────────────────────────────────────────────────────


async def _populate_activity(
    activity: Activity,
    src: ActivitySource,
    norm,
    client,
    access_token: str,
    athlete: Athlete,
    session: AsyncSession,
    *,
    user_id: str,
    prefetched_fit=_NOTFETCHED,
) -> None:
    """Populate a new Activity's metrics, streams and bests from src's data."""
    await _fill_from_source(activity, src, norm, client, access_token, athlete, session, user_id=user_id, prefetched_fit=prefetched_fit)


async def _repopulate_activity(
    activity: Activity,
    new_src: ActivitySource,
    norm,
    client,
    access_token: str,
    athlete: Athlete,
    session: AsyncSession,
    *,
    user_id: str,
    prefetched_fit=_NOTFETCHED,
) -> None:
    """Re-populate an existing Activity's metrics with data from a higher-priority source.

    Deletes all existing streams and bests first, then re-fills from the new source.
    Pass prefetched_fit to avoid downloading the FIT file twice (already fetched
    during the priority check in sync_provider_activities).
    """
    await session.execute(
        delete(ActivityStream).where(ActivityStream.activity_id == activity.id)
    )
    await session.execute(
        delete(ActivityPowerBest).where(ActivityPowerBest.activity_id == activity.id)
    )
    await session.execute(
        delete(ActivityDistanceBest).where(
            ActivityDistanceBest.activity_id == activity.id
        )
    )
    await session.execute(
        delete(ActivityInterval).where(ActivityInterval.activity_id == activity.id)
    )
    await session.flush()
    await _fill_from_source(
        activity,
        new_src,
        norm,
        client,
        access_token,
        athlete,
        session,
        user_id=user_id,
        prefetched_fit=prefetched_fit,
    )


# The two NormalizedActivity fields that aren't Activity columns. The rest of
# the provider-agnostic shape exists precisely so the summary copy is verbatim.
_NORM_ONLY_FIELDS = frozenset({"external_id", "source"})


def _summary_fields(activity: Activity, norm: NormalizedActivity) -> dict:
    """Activity fields taken straight off the provider's summary payload."""
    fields = {
        f.name: getattr(norm, f.name)
        for f in dataclasses.fields(norm)
        if f.name not in _NORM_ONLY_FIELDS
    }
    # A name or sport already on the row came from a higher-priority source.
    fields["name"] = activity.name or norm.name
    fields["sport_type"] = activity.sport_type or norm.sport_type
    return fields


async def _apply_import(
    activity: Activity,
    athlete: Athlete,
    session: AsyncSession,
    *,
    fields: dict,
    streams: dict[str, list],
    weight_log,
    load_duration_s: int,
) -> None:
    """Write imported metrics onto the Activity and persist the derived rows.

    Shared by the FIT and stream-API import paths, so a ride's stored metrics
    never depend on which provider won the priority contest.

    ``load_duration_s`` is separate from ``fields["duration_s"]`` because the FIT
    path falls back to the provider's duration for the load maths when the file
    has none, while the row still records what the file itself said.
    """
    power_data = streams.get("power") or []
    np_val = weighted_power(power_data) if power_data else None
    load, intensity = calculate_load(
        load_duration_s, np_val, fields.get("avg_hr"), athlete.ftp, athlete.max_hr
    )

    for key, value in fields.items():
        setattr(activity, key, value)
    activity.weighted_power = np_val
    activity.load = load
    activity.intensity = intensity
    activity.status = "processed"

    category = classify_workout(
        intensity, variability_index(np_val, activity.avg_power)
    )
    activity.workout_category = category.value if category else None

    _add_streams(activity, session, streams)
    _add_power_bests(activity, athlete, session, power_data, weight_log)
    _add_distance_bests(activity, athlete, session, streams.get("speed") or [])
    await _apply_aerobic(activity, athlete, session, streams)


async def _fill_from_source(
    activity: Activity,
    src: ActivitySource,
    norm,
    client,
    access_token: str,
    athlete: Athlete,
    session: AsyncSession,
    *,
    user_id: str,
    prefetched_fit=_NOTFETCHED,
) -> None:
    """Core import logic: try FIT first, fall back to stream API.

    prefetched_fit: if _NOTFETCHED, the FIT will be downloaded here.
                    If None, FIT was already tried and failed (skip download).
                    If bytes, use the pre-fetched FIT data directly.
    """
    # Effective weight at the activity's date, for the W/kg on each power best.
    weight_log = await load_weight_log(athlete.id, session)

    # ── FIT-first path (Wahoo and any future FIT-capable provider) ──────
    if prefetched_fit is _NOTFETCHED:
        fit_bytes: bytes | None = None
        try:
            fit_bytes = await client.download_fit_file(access_token, norm.external_id)
        except Exception:
            fit_bytes = None
    else:
        fit_bytes = prefetched_fit  # type: ignore[assignment]

    if fit_bytes is not None:
        storage_dir = settings.user_fit_dir(athlete.global_user_id)
        storage_dir.mkdir(parents=True, exist_ok=True)
        fit_path = storage_dir / f"{activity.id}.fit"
        fit_path.write_bytes(fit_bytes)

        try:
            # In a thread: the same whole-file iteration the upload path does,
            # and a provider backfill runs it once per activity (#101 §2.2).
            profile = await asyncio.to_thread(summarizeWorkout, io.BytesIO(fit_bytes))
        except Exception:
            log.exception("FIT parsing failed for %s/%s", norm.source, norm.external_id)
            profile = None

        encrypted = False
        try:
            encrypt_file(fit_path, athlete.global_user_id)
            encrypted = True
        except Exception:
            log.warning("FIT encryption failed for activity %s", activity.id)

        src.fit_file_path = str(fit_path)
        src.fit_file_encrypted = encrypted

        if profile is not None:
            # Resampled onto the shared 1 Hz clock by the parser, gaps as None
            # (issue #76) — the same shape the upload path stores.
            streams = {
                "power": to_json_stream(profile.power),
                "heartrate": to_json_stream(profile.heartRate),
                "cadence": to_json_stream(profile.cadence),
                "speed": to_json_stream(
                    [None if v is None else v / 3.6 for v in profile.speed]
                ),
                "altitude": to_json_stream(profile.altitude),
            }
            has_power = bool(streams["power"])
            has_hr = bool(streams["heartrate"])

            # The device file wins on every field it can speak to; each falls
            # through to the provider summary when the file has nothing.
            fields = {
                "name": activity.name or norm.name or "Uploaded Activity",
                "sport_type": (
                    activity.sport_type
                    or norm.sport_type
                    or resolve_sport_type(profile.sport_type)
                ),
                "start_time": profile.start_time or norm.start_time,
                "duration_s": profile.duration,
                "distance_m": (
                    float(profile.distance) if profile.distance else norm.distance_m
                ),
                "elevation_m": (
                    float(profile.elevationGain)
                    if profile.elevationGain
                    else norm.elevation_m
                ),
                "avg_power": profile.avgPower if has_power else norm.avg_power,
                "avg_hr": profile.avgHeartRate if has_hr else norm.avg_hr,
                "max_hr": profile.peakHR if has_hr else norm.max_hr,
                "avg_speed_ms": (
                    (profile.avgSpeed / 3.6) if profile.speed else norm.avg_speed_ms
                ),
                "avg_cadence": (
                    float(profile.avgCadence) if profile.cadence else norm.avg_cadence
                ),
            }

            await _apply_import(
                activity, athlete, session, fields=fields, streams=streams,
                weight_log=weight_log,
                load_duration_s=profile.duration or norm.duration_s or 0,
            )
            # After _apply_import, so the laps see the w_bal stream it derived.
            await rebuild_intervals(
                activity, session, io.BytesIO(fit_bytes), streams
            )
        else:
            # FIT parse failed — summary metadata only. No streams means no load,
            # intensity or category; a reprocess can still fill them in later.
            for key, value in _summary_fields(activity, norm).items():
                setattr(activity, key, value)
            activity.status = "processed"

        await session.commit()
        await session.refresh(activity)
        return

    # ── Stream-based fallback (Strava, providers without FIT download) ───
    try:
        streams_raw = await client.get_activity_streams(access_token, norm.external_id)
    except Exception:
        streams_raw = {}

    # The provider client is responsible for putting its streams on the shared
    # 1 Hz clock before they get here (see ``providers.strava``); this only
    # normalises which keys exist.
    streams = {
        key: to_json_stream(streams_raw.get(key, []))
        for key in ("power", "heartrate", "cadence", "speed", "altitude")
    }
    power_data = [v for v in streams["power"] if v is not None]
    hr_data = [v for v in streams["heartrate"] if v is not None]

    await _apply_import(
        activity, athlete, session,
        # Only the two averages the streams can improve on differ from the
        # summary; everything else the provider told us stands.
        fields=_summary_fields(activity, norm) | {
            "avg_power": norm.avg_power or (
                sum(power_data) / len(power_data) if power_data else None
            ),
            "avg_hr": (sum(hr_data) / len(hr_data)) if hr_data else norm.avg_hr,
        },
        streams=streams,
        weight_log=weight_log,
        load_duration_s=norm.duration_s or 0,
    )

    await session.commit()
    await session.refresh(activity)


# ── Stream / bests helpers ────────────────────────────────────────────────────


async def _apply_aerobic(
    activity: Activity,
    athlete: Athlete,
    session: AsyncSession,
    stream_map: dict[str, list],
) -> None:
    """Derive the aerobic metrics and persist the W' balance stream.

    Both provider paths need this, and so do the upload and reprocess paths in
    the API layer — see ``services.aerobic_metrics`` for why it lives in one
    place. Must run after ``_add_power_bests``: autoflush is what lets the
    date-restricted CP fit see the ride's own efforts.
    """
    w_bal_data = await apply_aerobic_metrics(activity, athlete, stream_map, session)
    if not w_bal_data:
        return
    stream_map["w_bal"] = w_bal_data
    session.add(
        ActivityStream(
            id=str(uuid.uuid4()),
            activity_id=activity.id,
            stream_type="w_bal",
            data=w_bal_data,
        )
    )


def _add_streams(
    activity: Activity, session: AsyncSession, streams: dict[str, list]
) -> None:
    """Persist the recorded streams, plus the torque derived from power+cadence."""
    derived = {
        "torque": compute_torque_stream(
            streams.get("power") or [], streams.get("cadence") or []
        )
    }
    for stream_type, data in {**streams, **derived}.items():
        if data:
            session.add(
                ActivityStream(
                    id=str(uuid.uuid4()),
                    activity_id=activity.id,
                    stream_type=stream_type,
                    data=data,
                )
            )


def _add_power_bests(
    activity: Activity,
    athlete: Athlete,
    session: AsyncSession,
    power_data: list,
    weight_log: list[tuple[date, float]] | None = None,
    *,
    weight: float | None = None,
) -> None:
    """Insert this activity's peak-power rows, each stamped with a bodyweight.

    ``weight`` pins the bodyweight instead of reading it from ``weight_log``.
    The reprocess path passes the weight already stored on the existing rows, so
    rebuilding never re-attributes an old effort to a weight the athlete only
    logged later.
    """
    if not power_data:
        return
    if weight is None:
        act_date = activity.start_time.date() if activity.start_time else None
        weight = effective_weight_for(weight_log or [], act_date)
    for dur_s, pwr_w in compute_power_bests(power_data).items():
        session.add(
            ActivityPowerBest(
                activity_id=activity.id,
                athlete_id=athlete.id,
                duration_s=dur_s,
                power_w=pwr_w,
                activity_start_time=activity.start_time,
                weight_kg=weight,
                w_per_kg=w_per_kg(pwr_w, weight),
            )
        )


def _add_distance_bests(
    activity: Activity,
    athlete: Athlete,
    session: AsyncSession,
    speed_data: list,
) -> None:
    if not speed_data:
        return
    for dist_m, time_s in compute_distance_bests(speed_data).items():
        session.add(
            ActivityDistanceBest(
                activity_id=activity.id,
                athlete_id=athlete.id,
                distance_m=dist_m,
                time_s=time_s,
                activity_start_time=activity.start_time,
            )
        )


async def rebuild_intervals(
    activity: Activity,
    session: AsyncSession,
    fileish,
    stream_map: dict[str, list],
    *,
    replace: bool = False,
    fmt: str = "fit",
) -> None:
    """Give this activity its interval breakdown, from recorded laps or an auto-split.

    ``fileish`` is anything the format's ``extractIntervals`` accepts — a path, a
    file object, or ``None`` when there is no original file to read. A file with
    no usable lap records (or no file at all) falls back to fixed-length
    auto-splits sized by the ride duration, so every processed activity ends up
    with a breakdown.

    ``fmt`` names which parser to read the laps with (issue #36). GPX has no lap
    concept and always auto-splits; TCX carries the athlete's own splits, same as
    FIT.

    ``replace=True`` clears the existing rows first, for the reprocess and
    attach-a-FIT paths that run against an activity which already has intervals;
    the import path is populating a fresh one and leaves it off.
    """
    if replace:
        await session.execute(
            delete(ActivityInterval).where(ActivityInterval.activity_id == activity.id)
        )
        await session.flush()

    raw = (
        await asyncio.to_thread(parser_for(fmt).extractIntervals, fileish)
        if fileish is not None
        else []
    )
    is_auto = len(raw) <= 1
    if is_auto:
        # A stream can outrun the recorded duration; split across the longer of
        # the two so the tail of the ride isn't dropped from the breakdown.
        stream_length = max((len(v) for v in stream_map.values() if v), default=0)
        duration_s = max(activity.duration_s or 0, stream_length)
        if duration_s and activity.start_time:
            raw = build_auto_intervals(
                activity.start_time, duration_s, auto_interval_s(duration_s)
            )

    if raw and activity.start_time:
        for iv in compute_interval_stats(
            raw, activity.start_time, stream_map, is_auto
        ):
            session.add(
                ActivityInterval(id=str(uuid.uuid4()), activity_id=activity.id, **iv)
            )
