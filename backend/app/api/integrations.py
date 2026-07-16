"""
Generic provider integration routes.

Handles OAuth connect/callback, sync, and disconnect for all registered
providers (Strava, Wahoo, …). Adding a new provider requires only registering
it in providers/registry.py — no new router code needed.

ProviderConnection records live in the registry DB (global per-user). Activity
data is written to the user's own DB on sync.
"""

import logging
from datetime import date, datetime, timezone

import httpx
from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.consent import require_consent
from backend.app.core.config import settings
from backend.app.core.deps import get_ctx_and_session
from backend.app.db.registry import get_registry_session
from backend.app.models.registry_orm import ProviderConnection
from backend.app.models.user_orm import Activity, ActivitySource, Athlete
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


def _encode_state(user_id: str, provider: str) -> str:
    return jwt.encode(
        {"sub": user_id, "purpose": f"{provider}_oauth"},
        settings.secret_key,
        algorithm="HS256",
    )


def _decode_state(state: str, provider: str) -> str:
    """Decode a state JWT and return the user_id. Raises JWTError on failure."""
    payload = jwt.decode(state, settings.secret_key, algorithms=["HS256"])
    if payload.get("purpose") != f"{provider}_oauth":
        raise JWTError("wrong purpose")
    return payload["sub"]


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

@router.get("/{provider}/connect", dependencies=[Depends(require_consent)])
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

    state = _encode_state(ctx.user_id, provider)
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
        user_id = _decode_state(state, provider)
    except (JWTError, KeyError, ValueError):
        return RedirectResponse(
            url=f"{settings.frontend_url}?{provider}=error"
        )

    redirect_base = settings.frontend_url

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

    Imports the full history into the user's own DB.
    """
    ctx, _ = ctx_session
    _require_provider(provider)
    await _get_connection(ctx.user_id, provider, registry_session)  # ensure connected
    await registry_session.close()  # return pool connection before bg task needs its own
    background_tasks.add_task(_bg_provider_sync, ctx.user_id, provider)
    return {"status": "sync started"}


async def _bg_provider_sync(user_id: str, provider: str) -> None:
    from backend.app.db.registry import _RegistrySessionLocal
    from backend.app.db.user_session import get_user_session_factory, init_user_db
    from backend.app.services.metrics_engine import recalculate_from
    from backend.app.services.weight import recompute_power_best_weights

    # Step 1: Refresh the token once from the registry.
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

    # Step 2: Sync into the user's own DB.
    try:
        await init_user_db(user_id)
        async with get_user_session_factory(user_id)() as session:
            athlete_result = await session.execute(
                select(Athlete).where(Athlete.global_user_id == user_id)
            )
            athlete = athlete_result.scalar_one_or_none()
            if athlete is None:
                return

            count, earliest = await sync_provider_activities(
                athlete, conn, session, user_id=user_id, access_token=access_token
            )
            if count > 0 and earliest is not None:
                await recalculate_from(athlete.id, earliest, session)
                # Re-derive W/kg on every power best now that the full history is
                # imported (guards against any row created without weight data).
                await recompute_power_best_weights(athlete.id, session)
                await session.commit()

            log.info(
                "%s sync complete: %d new activities for user %s",
                provider, count, user_id,
            )
    except Exception:
        log.exception("%s sync failed for user %s", provider, user_id)


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


class DisconnectRequest(BaseModel):
    """Optional request body for the disconnect endpoint.

    ``delete_data`` may be supplied either as a query parameter or in this body;
    accepting both means a query-vs-body mismatch from the caller can't silently
    default to ``False`` and skip the deletion.
    """

    delete_data: bool = False


async def _delete_provider_data(user_id: str, provider: str) -> None:
    """Permanently delete every activity imported from ``provider``.

    Opens the user's DB, removes each ``ActivitySource`` for the provider (and
    any parent ``Activity`` that is left with no remaining sources), recalculates
    affected daily metrics, and commits. Raises on any failure so the caller can
    surface it — data deletion must never be reported as successful unless it was
    actually committed.
    """
    from pathlib import Path

    from backend.app.db.user_session import get_user_session_factory
    from backend.app.services.metrics_engine import recalculate_from

    async with get_user_session_factory(user_id)() as user_session:
        athlete_result = await user_session.execute(
            select(Athlete).where(Athlete.global_user_id == user_id)
        )
        athlete = athlete_result.scalar_one_or_none()
        if athlete is None:
            return

        src_result = await user_session.execute(
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
                Path(src.fit_file_path).unlink(missing_ok=True)
            await user_session.delete(src)
            await user_session.flush()

            remaining = await user_session.execute(
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
                await user_session.delete(act)
                await user_session.flush()

        if earliest_date is not None:
            await recalculate_from(athlete.id, earliest_date, user_session)

        await user_session.commit()


@router.delete("/{provider}/disconnect", status_code=204)
async def disconnect(
    provider: str,
    delete_data: bool = Query(False, description="Also delete all activities imported from this provider"),
    body: DisconnectRequest | None = Body(None),
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    """Revoke the provider token and remove the stored connection.

    Pass ``delete_data=true`` (as a query parameter or in the request body) to
    also permanently delete every activity imported from this provider from the
    user's data. If that deletion fails, the connection is left in place and the
    request fails with ``500`` — the caller is never told the data is gone unless
    it actually was.
    """
    ctx, _ = ctx_session
    _require_provider(provider)
    conn = await _get_connection(ctx.user_id, provider, registry_session)

    should_delete = delete_data or (body is not None and body.delete_data)

    if conn.access_token:
        try:
            client_cls = PROVIDERS[provider]
            await client_cls.deauthorize(conn.access_token)  # type: ignore[call-arg]
        except Exception:
            pass  # best-effort

    # Delete the data (and commit it) *before* removing the connection, so the
    # connection — and the 204 that signals success — only go away once the data
    # is really gone. A failure here must be loud, not swallowed.
    if should_delete:
        try:
            await _delete_provider_data(ctx.user_id, provider)
        except Exception:
            log.exception("Failed to delete %s data for user %s", provider, ctx.user_id)
            raise HTTPException(
                status_code=500,
                detail="Failed to delete activity data; the connection was left in place.",
            )

    await registry_session.delete(conn)
    await registry_session.commit()
