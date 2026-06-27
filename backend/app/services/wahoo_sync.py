"""
Wahoo webhook event processing.

Handles incoming workout_summary events posted directly to the main backend.
Full activity sync is handled by the generic provider_sync.py pipeline.

process_wahoo_webhook opens its own registry + team sessions so it can be
called directly from the webhook handler without a pre-existing session.
"""

import asyncio
import logging

from sqlalchemy import select

from backend.app.models.registry_orm import ProviderConnection, TeamMembership
from backend.app.models.team_orm import Activity, ActivitySource, Athlete
from backend.app.services.provider_sync import (
    _DUPLICATE_WINDOW,
    _get_activity_lock,
    _populate_activity,
    _repopulate_activity,
    _winning_priority,
    _source_priority,
    ensure_fresh_token,
)
from backend.app.services.providers.wahoo import WahooClient, _normalize_workout

log = logging.getLogger(__name__)

_wahoo_client = WahooClient()


async def process_wahoo_webhook(payload: dict) -> None:
    """
    Handle a single Wahoo workout_summary webhook event, fanning out to all
    teams the user belongs to.

    Payload structure:
    {
        "event_type": "workout_summary",
        "webhook_token": "<configured token>",
        "user": {"id": <wahoo_user_id>},
        "workout_summary": {...},
        "workout": {"id": ..., "starts": ..., "workout_type_id": ..., "workout_summary": {...}}
    }
    """
    from backend.app.db.registry import _RegistrySessionLocal
    from backend.app.db.team_session import get_team_session_factory

    wahoo_user_id = str(payload.get("user", {}).get("id", ""))
    if not wahoo_user_id:
        log.warning("Wahoo webhook missing user.id — ignoring")
        return

    workout_summary = payload.get("workout_summary") or {}
    workout = dict(payload.get("workout") or workout_summary.get("workout") or {})
    if not workout:
        log.warning("Wahoo webhook missing workout object — ignoring")
        return
    if not workout.get("workout_summary"):
        workout["workout_summary"] = workout_summary

    norm = _normalize_workout(workout)

    # Extract the CDN FIT URL attached to the webhook payload. This is the
    # original device file; prefer it over the API endpoint which may return a
    # processed version without structured-workout lap records.
    workout_summary_for_url = workout.get("workout_summary") or workout_summary
    cdn_fit_url: str | None = (workout_summary_for_url.get("file") or {}).get("url")

    # Resolve user and team memberships from registry
    async with _RegistrySessionLocal() as reg_session:
        conn_result = await reg_session.execute(
            select(ProviderConnection).where(
                ProviderConnection.provider == "wahoo",
                ProviderConnection.provider_athlete_id == wahoo_user_id,
            )
        )
        conn = conn_result.scalar_one_or_none()
        if conn is None:
            log.warning("Wahoo webhook for unknown user %s — ignoring", wahoo_user_id)
            return

        user_id = conn.user_id
        access_token = await ensure_fresh_token(conn, reg_session)

        mb_result = await reg_session.execute(
            select(TeamMembership).where(TeamMembership.user_id == user_id)
        )
        team_ids = [m.team_id for m in mb_result.scalars().all()]

    for team_id in team_ids:
        try:
            async with get_team_session_factory(team_id)() as session:
                athlete_result = await session.execute(
                    select(Athlete).where(Athlete.global_user_id == user_id)
                )
                athlete = athlete_result.scalar_one_or_none()
                if athlete is None:
                    continue

                await _process_wahoo_for_team(norm, athlete, conn, access_token, team_id, session, cdn_fit_url=cdn_fit_url)
        except Exception:
            log.exception(
                "Failed to process Wahoo webhook for user %s in team %s", user_id, team_id
            )


async def _download_fit_cdn_first(
    access_token: str, external_id: str, cdn_url: str | None
) -> bytes | None:
    """Download FIT bytes, preferring the CDN URL (original device file)
    over the API endpoint which may return a processed version."""
    import httpx
    if cdn_url:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=5.0), follow_redirects=True) as client:
                r = await client.get(cdn_url)
                if r.is_success:
                    return r.content
        except Exception:
            pass
    return await _wahoo_client.download_fit_file(access_token, external_id)


async def _process_wahoo_for_team(norm, athlete, conn, access_token, team_id, session, *, cdn_fit_url: str | None = None) -> None:
    from backend.app.services.metrics_engine import recalculate_from

    # Idempotent: skip if this (provider, external_id) is already imported
    dupe = await session.execute(
        select(ActivitySource)
        .join(Activity, ActivitySource.activity_id == Activity.id)
        .where(
            Activity.athlete_id == athlete.id,
            ActivitySource.provider == "wahoo",
            ActivitySource.external_id == norm.external_id,
        )
    )
    if dupe.scalar_one_or_none() is not None:
        log.debug("Wahoo webhook: activity %s already imported — skipping", norm.external_id)
        return

    async with _get_activity_lock(team_id, athlete.id):
        # Check for existing Activity at the same time window
        existing_result = await session.execute(
            select(Activity).where(
                Activity.athlete_id == athlete.id,
                Activity.start_time >= norm.start_time - _DUPLICATE_WINDOW,
                Activity.start_time <= norm.start_time + _DUPLICATE_WINDOW,
            )
        )
        existing_act = existing_result.scalar_one_or_none()

        # Guard: don't attach a second wahoo source to an activity that
        # already has one (two distinct Wahoo workouts close in time).
        if existing_act is not None and any(
            s.provider == "wahoo" for s in existing_act.sources
        ):
            existing_act = None

        if existing_act is not None:
            new_src = ActivitySource(
                activity_id=existing_act.id,
                provider="wahoo",
                external_id=norm.external_id,
            )
            session.add(new_src)
            await session.flush()

            prefetched_fit: bytes | None = None
            try:
                prefetched_fit = await _download_fit_cdn_first(
                    access_token, norm.external_id, cdn_fit_url
                )
            except Exception:
                prefetched_fit = None

            actual_priority = _source_priority("wahoo", prefetched_fit is not None)
            if actual_priority < _winning_priority(existing_act):
                await _repopulate_activity(
                    existing_act, new_src, norm, _wahoo_client, access_token,
                    athlete, session, team_id=team_id, prefetched_fit=prefetched_fit,
                )
                if existing_act.start_time:
                    start_date = (
                        existing_act.start_time.date()
                        if hasattr(existing_act.start_time, "date")
                        else existing_act.start_time
                    )
                    await recalculate_from(athlete.id, start_date, session)
            else:
                await session.commit()
            return

        # New workout — create Activity + ActivitySource
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
            provider="wahoo",
            external_id=norm.external_id,
        )
        session.add(src)
        await session.flush()

        # Commit inside the lock so the Activity is visible to concurrent
        # sessions before this lock is released (fixes the #76 race condition).
        await session.commit()

    prefetched_fit_new: bytes | None = None
    try:
        prefetched_fit_new = await _download_fit_cdn_first(
            access_token, norm.external_id, cdn_fit_url
        )
    except Exception:
        prefetched_fit_new = None

    await _populate_activity(
        activity, src, norm, _wahoo_client, access_token, athlete, session, team_id=team_id,
        prefetched_fit=prefetched_fit_new
    )

    if activity.start_time:
        start_date = (
            activity.start_time.date()
            if hasattr(activity.start_time, "date")
            else activity.start_time
        )
        await recalculate_from(athlete.id, start_date, session)

    app_cfg = athlete.app_settings or {}
    if app_cfg.get("auto_analyze"):
        from backend.app.services.llm_activity_analyzer import analyze_activity_bg
        activity.analysis_status = "pending"
        await session.commit()
        asyncio.create_task(analyze_activity_bg(activity.id, athlete.id, team_id))

    if app_cfg.get("auto_training_status") and athlete.training_status_status != "pending":
        from backend.app.services.llm_training_status_analyzer import analyze_training_status_bg
        from datetime import datetime, timezone
        athlete.training_status_status = "pending"
        athlete.training_status = None
        athlete.training_status_updated_at = datetime.now(timezone.utc)
        await session.commit()
        asyncio.create_task(analyze_training_status_bg(athlete.id, team_id))
