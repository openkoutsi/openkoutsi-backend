"""
Generic provider integration routes.

Handles OAuth connect/callback, sync, and disconnect for all registered
providers (Strava, Wahoo, …). Adding a new provider requires only registering
it in providers/registry.py — no new router code needed.

ProviderConnection records live in the registry DB (global per-user, not per-team).
Activity data is written to every team the user belongs to on sync.
"""

import logging
from datetime import date, datetime, timezone

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.deps import get_ctx_and_session
from backend.app.db.registry import get_registry_session
from backend.app.models.registry_orm import ProviderConnection, Team, TeamMembership
from backend.app.models.team_orm import Activity, ActivitySource, Athlete
from backend.app.services.provider_sync import ensure_fresh_token, sync_provider_activities
from backend.app.services.providers.registry import PROVIDERS

log = logging.getLogger(__name__)

router = APIRouter(prefix="/integrations", tags=["integrations"])


# ── Helpers ────────────────────────────────────────────────────────────────

def _require_provider(provider: str) -> type:
    client_cls = PROVIDERS.get(provider)
    if client_cls is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider}")
    return client_cls


async def _get_athlete(global_user_id: str, session: AsyncSession) -> Athlete:
    result = await session.execute(select(Athlete).where(Athlete.global_user_id == global_user_id))
    athlete = result.scalar_one_or_none()
    if athlete is None:
        raise HTTPException(status_code=404, detail="Athlete not found")
    return athlete


async def _get_connection(
    user_id: str, provider: str, session: AsyncSession
) -> ProviderConnection:
    result = await session.execute(
        select(ProviderConnection).where(
            ProviderConnection.user_id == user_id,
            ProviderConnection.provider == provider,
        )
    )
    conn = result.scalar_one_or_none()
    if conn is None:
        raise HTTPException(status_code=400, detail=f"{provider} is not connected")
    return conn


def _encode_state(user_id: str, team_slug: str, provider: str) -> str:
    return jwt.encode(
        {"sub": user_id, "team_slug": team_slug, "purpose": f"{provider}_oauth"},
        settings.secret_key,
        algorithm="HS256",
    )


def _decode_state(state: str, provider: str) -> tuple[str, str]:
    """Decode a state JWT and return (user_id, team_slug). Raises JWTError on failure."""
    payload = jwt.decode(state, settings.secret_key, algorithms=["HS256"])
    if payload.get("purpose") != f"{provider}_oauth":
        raise JWTError("wrong purpose")
    return payload["sub"], payload.get("team_slug", "")


# ── Status ─────────────────────────────────────────────────────────────────

@router.get("/available")
async def available(ctx_session=Depends(get_ctx_and_session)):
    """Return the list of provider names that have credentials configured on the server."""
    configured = []
    if settings.strava_client_id:
        configured.append("strava")
    if settings.wahoo_client_id:
        configured.append("wahoo")
    return {"available": configured}


@router.get("/status")
async def status(
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    """Return the list of provider names the current user has connected."""
    ctx, _ = ctx_session
    result = await registry_session.execute(
        select(ProviderConnection).where(ProviderConnection.user_id == ctx.user_id)
    )
    connections = result.scalars().all()
    return {"connected": [c.provider for c in connections]}


# ── OAuth connect / callback ───────────────────────────────────────────────

@router.get("/{provider}/connect")
async def connect(
    provider: str,
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    """Return the OAuth authorization URL for the given provider."""
    ctx, _ = ctx_session
    client_cls = _require_provider(provider)

    if provider == "strava" and not settings.strava_client_id:
        raise HTTPException(status_code=501, detail="Strava is not configured")
    if provider == "wahoo" and not settings.wahoo_client_id:
        raise HTTPException(status_code=501, detail="Wahoo is not configured")

    # Look up team slug so the callback can redirect back to the right team
    team_result = await registry_session.execute(
        select(Team).where(Team.id == ctx.team_id)
    )
    team = team_result.scalar_one()

    state = _encode_state(ctx.user_id, team.slug, provider)
    redirect_uri = f"{settings.api_url}/api/integrations/{provider}/callback"
    client = client_cls()
    url = client.get_oauth_url(state, redirect_uri)
    return {"url": url}


@router.get("/{provider}/callback")
async def callback(
    provider: str,
    code: str,
    state: str,
    registry_session: AsyncSession = Depends(get_registry_session),
):
    """Exchange OAuth code for tokens and persist the connection.

    Strava / Wahoo redirect the user's browser here, so Bearer auth is not
    available. We identify the user via the signed JWT in the ``state`` param.
    """
    client_cls = _require_provider(provider)

    try:
        user_id, team_slug = _decode_state(state, provider)
    except (JWTError, KeyError, ValueError):
        return RedirectResponse(
            url=f"{settings.frontend_url}?{provider}=error"
        )

    redirect_base = (
        f"{settings.frontend_url}/t/{team_slug}" if team_slug else settings.frontend_url
    )

    redirect_uri = f"{settings.api_url}/api/integrations/{provider}/callback"
    try:
        tokens = await client_cls.exchange_code(code, redirect_uri)  # type: ignore[call-arg]
    except httpx.HTTPStatusError:
        log.exception("%s code exchange failed", provider)
        return RedirectResponse(url=f"{redirect_base}/profile?{provider}=error")

    # Upsert ProviderConnection in registry (keyed by user_id + provider)
    conn_result = await registry_session.execute(
        select(ProviderConnection).where(
            ProviderConnection.user_id == user_id,
            ProviderConnection.provider == provider,
        )
    )
    conn = conn_result.scalar_one_or_none()
    if conn is None:
        conn = ProviderConnection(user_id=user_id, provider=provider)
        registry_session.add(conn)

    conn.provider_athlete_id = tokens.get("provider_athlete_id") or conn.provider_athlete_id
    conn.access_token = tokens["access_token"]
    conn.refresh_token = tokens["refresh_token"]
    conn.token_expires_at = datetime.fromtimestamp(
        tokens["expires_at"], tz=timezone.utc
    )
    await registry_session.commit()

    return RedirectResponse(url=f"{redirect_base}/profile?{provider}=connected")


# ── Sync ───────────────────────────────────────────────────────────────────

@router.post("/{provider}/sync")
async def sync(
    provider: str,
    background_tasks: BackgroundTasks,
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    """Trigger a full history import from the given provider in the background.

    Syncs to all teams the user belongs to.
    """
    ctx, _ = ctx_session
    _require_provider(provider)
    await _get_connection(ctx.user_id, provider, registry_session)  # ensure connected
    await registry_session.close()  # return pool connection before bg task needs its own
    background_tasks.add_task(_bg_provider_sync, ctx.user_id, provider)
    return {"status": "sync started"}


async def _bg_provider_sync(user_id: str, provider: str) -> None:
    from backend.app.db.registry import _RegistrySessionLocal
    from backend.app.db.team_session import get_team_session_factory
    from backend.app.services.metrics_engine import recalculate_from

    # Step 1: Refresh token once from registry, collect team membership list
    async with _RegistrySessionLocal() as reg_session:
        conn_result = await reg_session.execute(
            select(ProviderConnection).where(
                ProviderConnection.user_id == user_id,
                ProviderConnection.provider == provider,
            )
        )
        conn = conn_result.scalar_one_or_none()
        if conn is None:
            log.warning("No connection for user %s / provider %s", user_id, provider)
            return

        access_token = await ensure_fresh_token(conn, reg_session)

        mb_result = await reg_session.execute(
            select(TeamMembership).where(TeamMembership.user_id == user_id)
        )
        team_ids = [m.team_id for m in mb_result.scalars().all()]

    if not team_ids:
        log.warning("User %s has no team memberships — sync dropped", user_id)
        return

    # Step 2: Sync to each team
    for team_id in team_ids:
        try:
            async with get_team_session_factory(team_id)() as team_session:
                athlete_result = await team_session.execute(
                    select(Athlete).where(Athlete.global_user_id == user_id)
                )
                athlete = athlete_result.scalar_one_or_none()
                if athlete is None:
                    continue

                count, earliest = await sync_provider_activities(
                    athlete, conn, team_session, team_id=team_id, access_token=access_token
                )
                if count > 0 and earliest is not None:
                    await recalculate_from(athlete.id, earliest, team_session)

                log.info(
                    "%s sync complete: %d new activities for user %s in team %s",
                    provider, count, user_id, team_id,
                )
        except Exception:
            log.exception("%s sync failed for user %s in team %s", provider, user_id, team_id)


# ── Zone sync ──────────────────────────────────────────────────────────────

@router.post("/{provider}/sync-zones")
async def sync_zones(
    provider: str,
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    """Fetch training zones (HR, power) and FTP from the provider and save to the athlete profile."""
    ctx, session = ctx_session
    client_cls = _require_provider(provider)
    athlete = await _get_athlete(ctx.user_id, session)
    conn = await _get_connection(ctx.user_id, provider, registry_session)

    access_token = await ensure_fresh_token(conn, registry_session)

    client = client_cls()
    try:
        zone_data = await client.fetch_zones(access_token)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (401, 403):
            raise HTTPException(status_code=403, detail="insufficient_scope")
        raise HTTPException(status_code=502, detail="Provider API error during zone fetch")

    if zone_data is None:
        raise HTTPException(status_code=400, detail=f"{provider} does not support zone sync")

    if zone_data.ftp is None and zone_data.hr_zones is None and zone_data.power_zones is None:
        raise HTTPException(status_code=422, detail="no_zones_returned")

    updated: list[str] = []

    if zone_data.ftp is not None:
        if zone_data.ftp != athlete.ftp:
            athlete.ftp = zone_data.ftp
            ftp_tests = list(athlete.ftp_tests or [])
            ftp_tests.append({
                "date": date.today().isoformat(),
                "ftp": zone_data.ftp,
                "method": provider,
            })
            athlete.ftp_tests = ftp_tests
        updated.append("ftp")

    if zone_data.hr_zones is not None:
        athlete.hr_zones = zone_data.hr_zones
        updated.append("hr_zones")

    if zone_data.power_zones is not None:
        athlete.power_zones = zone_data.power_zones
        updated.append("power_zones")

    await session.commit()

    return {
        "updated": updated,
        "ftp": athlete.ftp,
        "hr_zones": athlete.hr_zones,
        "power_zones": athlete.power_zones,
    }


# ── Disconnect ─────────────────────────────────────────────────────────────

@router.delete("/{provider}/disconnect", status_code=204)
async def disconnect(
    provider: str,
    delete_data: bool = Query(False, description="Also delete all activities imported from this provider"),
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    """Revoke the provider token and remove the stored connection.

    Pass ``delete_data=true`` to also permanently delete all activities that
    were imported from this provider across all teams the user belongs to.
    """
    ctx, _ = ctx_session
    _require_provider(provider)
    conn = await _get_connection(ctx.user_id, provider, registry_session)

    if conn.access_token:
        try:
            client_cls = PROVIDERS[provider]
            await client_cls.deauthorize(conn.access_token)  # type: ignore[call-arg]
        except Exception:
            pass  # best-effort

    if delete_data:
        from pathlib import Path
        from backend.app.db.team_session import get_team_session_factory
        from backend.app.services.metrics_engine import recalculate_from

        mb_result = await registry_session.execute(
            select(TeamMembership).where(TeamMembership.user_id == ctx.user_id)
        )
        team_ids = [m.team_id for m in mb_result.scalars().all()]

        for team_id in team_ids:
            try:
                async with get_team_session_factory(team_id)() as team_session:
                    athlete_result = await team_session.execute(
                        select(Athlete).where(Athlete.global_user_id == ctx.user_id)
                    )
                    athlete = athlete_result.scalar_one_or_none()
                    if athlete is None:
                        continue

                    src_result = await team_session.execute(
                        select(ActivitySource)
                        .join(Activity, ActivitySource.activity_id == Activity.id)
                        .where(
                            Activity.athlete_id == athlete.id,
                            ActivitySource.provider == provider,
                        )
                    )
                    sources = src_result.scalars().all()

                    earliest_date = None
                    for src in sources:
                        act = src.activity
                        if src.fit_file_path:
                            p = Path(src.fit_file_path)
                            if p.exists():
                                p.unlink(missing_ok=True)
                        await team_session.delete(src)
                        await team_session.flush()

                        remaining = await team_session.execute(
                            select(ActivitySource).where(ActivitySource.activity_id == act.id)
                        )
                        if remaining.scalar_one_or_none() is None:
                            if act.start_time:
                                day = (
                                    act.start_time.date()
                                    if hasattr(act.start_time, "date")
                                    else act.start_time
                                )
                                if earliest_date is None or day < earliest_date:
                                    earliest_date = day
                            await team_session.delete(act)
                            await team_session.flush()

                    if earliest_date is not None:
                        await recalculate_from(athlete.id, earliest_date, team_session)

                    await team_session.commit()
            except Exception:
                log.exception("Failed to delete %s data for user %s in team %s", provider, ctx.user_id, team_id)

    await registry_session.delete(conn)
    await registry_session.commit()
