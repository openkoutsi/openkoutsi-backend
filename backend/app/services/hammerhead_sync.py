"""
Hammerhead webhook event processing (issue #69).

The bridge (``hammerhead_bridge/``) receives the webhook and queues it; the
poller in ``api/hammerhead.py`` claims events and hands each payload here. Full
history import is the generic ``provider_sync`` pipeline's job.

The payload is only ``{activityId, userId}`` — no summary and no file URL,
unlike Wahoo's CDN link — so each event costs two authenticated calls, detail
and FIT, made through :meth:`HammerheadClient.complete_activity` exactly as the
sync makes them.

Hammerhead documents a "manual retry" that delivers the same ``activityId``
again, and the bridge itself is at-least-once, so everything here is
idempotent on ``(provider, external_id)``.
"""

import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import select

from backend.app.models.registry_orm import ProviderConnection
from backend.app.models.user_orm import Activity, ActivitySource, Athlete
from backend.app.services.api_usage import attribute_to_user
from backend.app.services.provider_sync import (
    _DUPLICATE_WINDOW,
    _discard_partial_import,
    _populate_activity,
    _prefetch_fit,
    _repopulate_activity,
    _source_priority,
    _winning_priority,
    activity_create_guard,
    ensure_fresh_token,
)
from backend.app.services.providers.base import NormalizedActivity
from backend.app.services.providers.hammerhead import HammerheadClient, ScopeNotGranted
from backend.app.services.providers.throttling import ProviderThrottled
from backend.app.services.stranded_runs import (
    begin_activity_analysis_run,
    begin_training_status_run,
)

log = logging.getLogger(__name__)

PROVIDER = "hammerhead"


def _aware(dt: datetime) -> datetime:
    """SQLite hands datetimes back naive; every one stored here is UTC."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _ids(payload: dict) -> tuple[str, str] | None:
    """``(activity_id, user_id)`` from a webhook payload, or None if malformed."""
    if not isinstance(payload, dict):
        return None
    activity_id = payload.get("activityId") or payload.get("activity_id")
    user_id = payload.get("userId") or payload.get("user_id")
    if not activity_id or not user_id:
        return None
    return str(activity_id), str(user_id)


async def process_hammerhead_webhook(payload: dict) -> None:
    """Import the one activity a Hammerhead webhook names.

    Returns quietly for a malformed payload or an athlete we do not know —
    neither gets better on redelivery. A throttle or a transport error raises,
    so the poller nacks the event and the bridge serves it again after a
    backoff: Hammerhead itself never redelivers, so the bridge's queue is the
    only second chance an event gets.

    Nothing stops two openkoutsi accounts linking the same Karoo account, so the
    event is imported for *every* matching connection. One user's failure does
    not cost the other theirs: each is tried, and the first error is raised
    afterwards so the event is retried — harmlessly for whoever already has it,
    since the import is idempotent.
    """
    from backend.app.db.registry import _RegistrySessionLocal

    ids = _ids(payload)
    if ids is None:
        log.warning("Hammerhead webhook without activityId/userId — ignoring")
        return
    activity_id, hh_user_id = ids

    async with _RegistrySessionLocal() as reg_session:
        conn_ids = (
            await reg_session.execute(
                select(ProviderConnection.id).where(
                    ProviderConnection.provider == PROVIDER,
                    ProviderConnection.provider_athlete_id == hh_user_id,
                )
            )
        ).scalars().all()
    if not conn_ids:
        log.warning("Hammerhead webhook for unknown user %s — ignoring", hh_user_id)
        return
    if len(conn_ids) > 1:
        log.warning(
            "Hammerhead user %s is linked to %d accounts — importing for each",
            hh_user_id,
            len(conn_ids),
        )

    first_error: Exception | None = None
    for conn_id in conn_ids:
        try:
            # A registry session per connection, closed before the import, as
            # the other webhook paths do: the import can wait on a lease.
            async with _RegistrySessionLocal() as reg_session:
                conn = await reg_session.get(ProviderConnection, conn_id)
                if conn is None:
                    continue
                user_id = conn.user_id
                access_token = await ensure_fresh_token(conn, reg_session)
            await _import_for_connection(activity_id, user_id, access_token)
        except Exception as exc:
            log.exception(
                "Hammerhead activity %s failed for connection %s", activity_id, conn_id
            )
            first_error = first_error or exc
    if first_error is not None:
        raise first_error


async def _import_for_connection(activity_id: str, user_id: str, access_token: str) -> None:
    from backend.app.db.user_session import get_user_session_factory, init_user_db

    await init_user_db(user_id)
    async with get_user_session_factory(user_id)() as session:
        athlete = (
            await session.execute(select(Athlete).where(Athlete.global_user_id == user_id))
        ).scalar_one_or_none()
        if athlete is None:
            return
        with attribute_to_user(user_id):
            await _import_for_user(
                activity_id, athlete, access_token, user_id, session,
                client=HammerheadClient(),
            )


async def _already_imported(session, athlete_id: str, external_id: str) -> bool:
    result = await session.execute(
        select(ActivitySource.id)
        .join(Activity, ActivitySource.activity_id == Activity.id)
        .where(
            Activity.athlete_id == athlete_id,
            ActivitySource.provider == PROVIDER,
            ActivitySource.external_id == external_id,
        )
    )
    return result.first() is not None


async def _import_for_user(
    activity_id: str,
    athlete: Athlete,
    access_token: str,
    user_id: str,
    session,
    *,
    client: HammerheadClient,
) -> None:
    from backend.app.services.achievements import mark_achievements_dirty
    from backend.app.services.metrics_engine import recalculate_from

    # Idempotent: the documented manual retry, or a bridge redelivery, of an
    # activity that is already here. Checked before any provider call.
    if await _already_imported(session, athlete.id, activity_id):
        log.debug("Hammerhead activity %s already imported — skipping", activity_id)
        return

    sparse = NormalizedActivity(
        external_id=activity_id, source=PROVIDER, name=None, sport_type=None,
        start_time=datetime.now(timezone.utc), duration_s=None, distance_m=None,
        elevation_m=None, avg_power=None, avg_hr=None, max_hr=None,
        avg_speed_ms=None, avg_cadence=None,
    )
    try:
        norm = await client.complete_activity(access_token, sparse)
    except LookupError:
        # No file to date it by — a real answer, and redelivery would get the
        # same one.
        log.warning("Hammerhead activity %s has no usable FIT — ignoring", activity_id)
        return
    except ScopeNotGranted:
        # The athlete withheld `activity:read`; no retry will change that.
        log.warning("Hammerhead activity %s: activity:read not granted — ignoring", activity_id)
        return

    async with activity_create_guard(session, user_id, athlete.id):
        # Re-checked under the guard: two deliveries of the same event can both
        # pass the cheap check above.
        if await _already_imported(session, athlete.id, activity_id):
            return

        # The window can hold more than one activity — a ride Strava split in
        # two lands as two — and raising on that would retire an event that
        # never comes back. So take every candidate that is not already a Karoo
        # ride, and attach to the closest.
        candidates = [
            act
            for act in (
                await session.execute(
                    select(Activity).where(
                        Activity.athlete_id == athlete.id,
                        Activity.start_time >= norm.start_time - _DUPLICATE_WINDOW,
                        Activity.start_time <= norm.start_time + _DUPLICATE_WINDOW,
                    )
                )
            ).scalars().all()
            if not any(src.provider == PROVIDER for src in act.sources)
        ]
        existing_act = min(
            candidates,
            key=lambda act: abs((_aware(act.start_time) - norm.start_time).total_seconds()),
            default=None,
        )

        if existing_act is not None:
            new_src = ActivitySource(
                activity_id=existing_act.id, provider=PROVIDER, external_id=activity_id
            )
            session.add(new_src)
            await session.flush()
            # Served from the client's cache: `complete_activity` fetched it.
            fit, fit_resolved = await _prefetch_fit(
                client, access_token, norm, provider_name=PROVIDER, may_wait=False
            )
            if _source_priority(PROVIDER, fit is not None) < _winning_priority(existing_act):
                await _repopulate_activity(
                    existing_act, new_src, norm, client, access_token, athlete,
                    session, user_id=user_id, prefetched_fit=fit,
                    prefetched_fit_resolved=fit_resolved,
                )
                if existing_act.start_time:
                    await recalculate_from(
                        athlete.id, existing_act.start_time.date(), session
                    )
            else:
                if fit_resolved:
                    new_src.streams_fetched_at = datetime.now(timezone.utc)
                await session.commit()
            await mark_achievements_dirty(athlete.id, session)
            return

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
            activity_id=activity.id, provider=PROVIDER, external_id=activity_id
        )
        session.add(src)
        await session.flush()
        # Committed inside the guard, so the next caller's window sees it.
        await session.commit()

    try:
        await _populate_activity(
            activity, src, norm, client, access_token, athlete, session, user_id=user_id
        )
    except ProviderThrottled:
        await _discard_partial_import(session, activity.id)
        raise

    if activity.start_time:
        await recalculate_from(athlete.id, activity.start_time.date(), session)

    from backend.app.services.activity_workout_matcher import find_and_link_workout
    from backend.app.services.plan_adherence import catch_up_adherence

    await find_and_link_workout(session, athlete.id, activity)
    await catch_up_adherence(athlete.id, session)
    await mark_achievements_dirty(athlete.id, session)

    app_cfg = athlete.app_settings or {}
    llm_ok = True
    if app_cfg.get("auto_analyze") or app_cfg.get("auto_training_status"):
        from backend.app.services.llm_access import auto_analysis_allowed

        llm_ok = await auto_analysis_allowed(user_id, athlete)
    if llm_ok and app_cfg.get("auto_analyze"):
        from backend.app.services.llm_activity_analyzer import analyze_activity_bg

        run_id = begin_activity_analysis_run(activity)
        await session.commit()
        asyncio.create_task(
            analyze_activity_bg(activity.id, athlete.id, user_id, run_id=run_id)
        )
    if (
        llm_ok
        and app_cfg.get("auto_training_status")
        and athlete.training_status_status != "pending"
    ):
        from backend.app.services.llm_training_status_analyzer import (
            analyze_training_status_bg,
        )

        status_run = begin_training_status_run(athlete)
        await session.commit()
        asyncio.create_task(
            analyze_training_status_bg(athlete.id, user_id, run_id=status_run)
        )
