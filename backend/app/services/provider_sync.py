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
import io
import logging
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.file_encryption import encrypt_file
from backend.app.models.registry_orm import ProviderConnection
from backend.app.models.team_orm import (
    Activity,
    ActivityDistanceBest,
    ActivityInterval,
    ActivityPowerBest,
    ActivitySource,
    ActivityStream,
    Athlete,
)
from openkoutsi.categorization import classify_workout
from openkoutsi.fit_processing import (
    resolve_sport_type,
    auto_interval_s,
    build_auto_intervals,
    compute_interval_stats,
)
from backend.app.services.providers.registry import PROVIDERS
from openkoutsi.training_math import (
    calculate_tss,
    compute_distance_bests,
    compute_power_bests,
    normalized_power,
)
from openkoutsi.fit import summarizeWorkout, extractIntervals

log = logging.getLogger(__name__)

_DUPLICATE_WINDOW = timedelta(minutes=5)

# Sentinel: _fill_from_source uses this to know FIT hasn't been fetched yet
_NOTFETCHED = object()

# Per-(team_id, athlete_id) lock that serialises the dedup-window-query +
# create/attach operation. Prevents the race condition where two concurrent
# syncs both see "no existing activity" and each create a new one for the
# same real-world workout.
_activity_creation_locks: dict[tuple[str, str], asyncio.Lock] = {}


def _get_activity_lock(team_id: str, athlete_id: str) -> asyncio.Lock:
    key = (team_id, athlete_id)
    if key not in _activity_creation_locks:
        _activity_creation_locks[key] = asyncio.Lock()
    return _activity_creation_locks[key]


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


async def ensure_fresh_token(conn: ProviderConnection, session: AsyncSession) -> str:
    """Refresh the access token if it will expire soon. Returns current token."""
    expires_at = conn.token_expires_at
    if expires_at is not None and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    lookahead = _REFRESH_LOOKAHEAD.get(conn.provider, _DEFAULT_REFRESH_LOOKAHEAD)
    if expires_at and datetime.now(timezone.utc) + lookahead >= expires_at and conn.refresh_token:
        client_cls = PROVIDERS.get(conn.provider)
        if client_cls is None:
            log.warning("Unknown provider %s — cannot refresh token", conn.provider)
            return conn.access_token or ""

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
        await session.commit()
        log.info("Refreshed %s token for user %s", conn.provider, conn.user_id)

    return conn.access_token or ""


# ── Full sync ─────────────────────────────────────────────────────────────────


async def sync_provider_activities(
    athlete: Athlete,
    connection: ProviderConnection,
    session: AsyncSession,
    *,
    team_id: str,
    access_token: str | None = None,
) -> tuple[int, date | None]:
    """
    Import all activities from a provider that aren't already in the database.

    For each activity from the provider:
      - If this (provider, external_id) pair already has an ActivitySource → skip.
      - If an Activity exists within ±5 min → attach a new ActivitySource and
        repopulate the Activity if the new source has higher priority.
      - Otherwise → create a new Activity + ActivitySource.

    Returns (count_created_or_updated, earliest_start_date).
    """
    provider_name = connection.provider
    client_cls = PROVIDERS.get(provider_name)
    if client_cls is None:
        log.error("No client registered for provider %s", provider_name)
        return 0, None

    if access_token is None:
        access_token = await ensure_fresh_token(connection, session)
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
                    if act.normalized_power and athlete.ftp:
                        new_tss, new_if = calculate_tss(
                            norm.duration_s,
                            act.normalized_power,
                            act.avg_hr,
                            athlete.ftp,
                            athlete.max_hr,
                        )
                        act.tss = new_tss
                        act.intensity_factor = new_if
                    elif act.avg_hr and athlete.max_hr:
                        new_tss, _ = calculate_tss(
                            norm.duration_s,
                            None,
                            act.avg_hr,
                            athlete.ftp,
                            athlete.max_hr,
                        )
                        act.tss = new_tss
                    await session.commit()
                    log.info(
                        "Corrected duration for %s/%s: %ds → %ds",
                        provider_name,
                        ext_id,
                        old_dur,
                        norm.duration_s,
                    )
                continue

            # ── Find-or-create under a per-athlete lock ───────────────────
            # The lock serialises the dedup window query + commit so that two
            # concurrent syncs (e.g. Wahoo webhook + Strava full sync firing
            # within milliseconds) cannot both see "no existing activity" and
            # each create a duplicate record.
            #
            # Critical invariant: the new Activity row must be COMMITTED before
            # this lock is released.  A flush alone is not sufficient — under
            # READ COMMITTED isolation (and SQLite WAL mode) another session
            # that acquires the lock after the flush but before the commit will
            # still see an empty dedup window and create a duplicate.
            async with _get_activity_lock(team_id, athlete.id):
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
                            team_id=team_id,
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
                activity, src, norm, client, access_token, athlete, session, team_id=team_id
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
                from backend.app.services.llm_activity_analyzer import (
                    analyze_activity_bg,
                )

                activity.analysis_status = "pending"
                await session.commit()
                asyncio.create_task(analyze_activity_bg(activity.id, athlete.id, team_id))

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
    team_id: str,
    prefetched_fit=_NOTFETCHED,
) -> None:
    """Populate a new Activity's metrics, streams and bests from src's data."""
    await _fill_from_source(activity, src, norm, client, access_token, athlete, session, team_id=team_id, prefetched_fit=prefetched_fit)


async def _repopulate_activity(
    activity: Activity,
    new_src: ActivitySource,
    norm,
    client,
    access_token: str,
    athlete: Athlete,
    session: AsyncSession,
    *,
    team_id: str,
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
        team_id=team_id,
        prefetched_fit=prefetched_fit,
    )


async def _fill_from_source(
    activity: Activity,
    src: ActivitySource,
    norm,
    client,
    access_token: str,
    athlete: Athlete,
    session: AsyncSession,
    *,
    team_id: str,
    prefetched_fit=_NOTFETCHED,
) -> None:
    """Core import logic: try FIT first, fall back to stream API.

    prefetched_fit: if _NOTFETCHED, the FIT will be downloaded here.
                    If None, FIT was already tried and failed (skip download).
                    If bytes, use the pre-fetched FIT data directly.
    """
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
        storage_dir = settings.team_fit_dir(team_id, athlete.global_user_id)
        storage_dir.mkdir(parents=True, exist_ok=True)
        fit_path = storage_dir / f"{activity.id}.fit"
        fit_path.write_bytes(fit_bytes)

        try:
            profile = summarizeWorkout(io.BytesIO(fit_bytes))
        except Exception:
            log.exception("FIT parsing failed for %s/%s", norm.source, norm.external_id)
            profile = None

        encrypted = False
        try:
            encrypt_file(fit_path, team_id, athlete.global_user_id)
            encrypted = True
        except Exception:
            log.warning("FIT encryption failed for activity %s", activity.id)

        src.fit_file_path = str(fit_path)
        src.fit_file_encrypted = encrypted

        if profile is not None:
            power_data = [float(v) for v in profile.power]
            hr_data = [float(v) for v in profile.heartRate]
            cadence_data = [float(v) for v in profile.cadence]
            speed_ms = [v / 3.6 for v in profile.speed]
            alt_data = [float(v) for v in profile.altitude]

            np_val = normalized_power(power_data) if power_data else None
            avg_hr_v = profile.avgHeartRate if hr_data else norm.avg_hr
            dur_v = profile.duration or norm.duration_s or 0
            tss, intensity_factor = calculate_tss(
                dur_v, np_val, avg_hr_v, athlete.ftp, athlete.max_hr
            )

            activity.name = activity.name or norm.name or "Uploaded Activity"
            activity.sport_type = (
                activity.sport_type
                or norm.sport_type
                or resolve_sport_type(profile.sport_type)
            )
            activity.start_time = profile.start_time or norm.start_time
            activity.duration_s = profile.duration
            activity.distance_m = (
                float(profile.distance) if profile.distance else norm.distance_m
            )
            activity.elevation_m = (
                float(profile.elevationGain)
                if profile.elevationGain
                else norm.elevation_m
            )
            activity.avg_power = profile.avgPower if power_data else norm.avg_power
            activity.normalized_power = np_val
            activity.avg_hr = avg_hr_v
            activity.max_hr = profile.peakHR if hr_data else norm.max_hr
            activity.avg_speed_ms = (
                (profile.avgSpeed / 3.6) if profile.speed else norm.avg_speed_ms
            )
            activity.avg_cadence = (
                float(profile.avgCadence) if profile.cadence else norm.avg_cadence
            )
            activity.tss = tss
            activity.intensity_factor = intensity_factor
            activity.status = "processed"

            vi = (np_val / activity.avg_power) if (np_val and activity.avg_power) else None
            category = classify_workout(intensity_factor, vi)
            activity.workout_category = category.value if category else None

            _add_streams(
                activity, session, power_data, hr_data, cadence_data, speed_ms, alt_data
            )
            _add_power_bests(activity, athlete, session, power_data)
            _add_distance_bests(activity, athlete, session, speed_ms)

            stream_map = {
                "power": power_data,
                "heartrate": hr_data,
                "cadence": cadence_data,
                "speed": speed_ms,
                "altitude": alt_data,
            }
            _add_intervals(activity, session, fit_bytes, profile.start_time, stream_map)
        else:
            # FIT parse failed — use summary metadata only
            activity.name = activity.name or norm.name
            activity.sport_type = activity.sport_type or norm.sport_type
            activity.start_time = norm.start_time
            activity.duration_s = norm.duration_s
            activity.distance_m = norm.distance_m
            activity.elevation_m = norm.elevation_m
            activity.avg_power = norm.avg_power
            activity.avg_hr = norm.avg_hr
            activity.max_hr = norm.max_hr
            activity.avg_speed_ms = norm.avg_speed_ms
            activity.avg_cadence = norm.avg_cadence
            activity.status = "processed"

        await session.commit()
        await session.refresh(activity)
        return

    # ── Stream-based fallback (Strava, providers without FIT download) ───
    try:
        streams_raw = await client.get_activity_streams(access_token, norm.external_id)
    except Exception:
        streams_raw = {}

    power_data = streams_raw.get("power", [])
    hr_data = streams_raw.get("heartrate", [])
    cadence_data = streams_raw.get("cadence", [])
    speed_data = streams_raw.get("speed", [])
    altitude_data = streams_raw.get("altitude", [])

    np_val = normalized_power(power_data) if power_data else None
    avg_hr = (sum(hr_data) / len(hr_data)) if hr_data else norm.avg_hr
    dur_s = norm.duration_s or 0
    tss, intensity_factor = calculate_tss(
        dur_s, np_val, avg_hr, athlete.ftp, athlete.max_hr
    )

    activity.name = activity.name or norm.name
    activity.sport_type = activity.sport_type or norm.sport_type
    activity.start_time = norm.start_time
    activity.duration_s = norm.duration_s
    activity.distance_m = norm.distance_m
    activity.elevation_m = norm.elevation_m
    activity.avg_power = norm.avg_power or (
        sum(power_data) / len(power_data) if power_data else None
    )
    activity.normalized_power = np_val
    activity.avg_hr = avg_hr
    activity.max_hr = norm.max_hr
    activity.avg_speed_ms = norm.avg_speed_ms
    activity.avg_cadence = norm.avg_cadence
    activity.tss = tss
    activity.intensity_factor = intensity_factor
    activity.status = "processed"

    vi = (np_val / activity.avg_power) if (np_val and activity.avg_power) else None
    category = classify_workout(intensity_factor, vi)
    activity.workout_category = category.value if category else None

    _add_streams(
        activity, session, power_data, hr_data, cadence_data, speed_data, altitude_data
    )
    _add_power_bests(activity, athlete, session, power_data)
    _add_distance_bests(activity, athlete, session, speed_data)

    await session.commit()
    await session.refresh(activity)


# ── Stream / bests helpers ────────────────────────────────────────────────────


def _add_streams(
    activity: Activity,
    session: AsyncSession,
    power_data: list,
    hr_data: list,
    cadence_data: list,
    speed_data: list,
    altitude_data: list,
) -> None:
    for stream_type, data in [
        ("power", power_data),
        ("heartrate", hr_data),
        ("cadence", cadence_data),
        ("speed", speed_data),
        ("altitude", altitude_data),
    ]:
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
) -> None:
    if not power_data:
        return
    for dur_s, pwr_w in compute_power_bests(power_data).items():
        session.add(
            ActivityPowerBest(
                activity_id=activity.id,
                athlete_id=athlete.id,
                duration_s=dur_s,
                power_w=pwr_w,
                activity_start_time=activity.start_time,
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


def _add_intervals(
    activity: Activity,
    session: AsyncSession,
    fit_bytes: bytes,
    activity_start: datetime,
    stream_map: dict,
) -> None:
    import io as _io
    raw = extractIntervals(_io.BytesIO(fit_bytes))
    is_auto = len(raw) <= 1
    if is_auto:
        duration_s = activity.duration_s or 0
        if duration_s:
            interval_s = auto_interval_s(duration_s)
            raw = build_auto_intervals(activity_start, duration_s, interval_s)
    if raw:
        for iv in compute_interval_stats(raw, activity_start, stream_map, is_auto):
            session.add(ActivityInterval(id=str(uuid.uuid4()), activity_id=activity.id, **iv))
