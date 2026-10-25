"""
Hammerhead-specific routes and background infrastructure (issue #69).

Connect / sync / disconnect go through the generic ``/integrations/*`` routes.
What lives here is what has no generic equivalent:

* ``POST /integrations/hammerhead/push-metrics`` — training metrics go *to* the
  Karoo. The API is write-only for them (no ``GET /metrics``), so the generic
  ``sync-zones`` pull does not apply.
* The Hammerhead bridge poller — claims queued webhook events from
  ``hammerhead_bridge/`` and imports each one.
"""

import asyncio
import contextlib
import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.deps import get_ctx_session_athlete
from backend.app.core.scopes import pat_scopes
from backend.app.db.registry import get_registry_session
from backend.app.models.registry_orm import ProviderConnection
from backend.app.services.bridge_client import BridgeClient
from backend.app.services.hammerhead_sync import process_hammerhead_webhook
from backend.app.services.provider_sync import ensure_fresh_token
from backend.app.services.providers.hammerhead import (
    HammerheadClient,
    ScopeNotGranted,
    build_metrics_payload,
)

log = logging.getLogger(__name__)

PROVIDER = "hammerhead"

#: The ``app_settings`` key that opts an athlete in to pushing metrics to the
#: Karoo whenever FTP, weight, heart rate or zones change. Off by default: it
#: writes to the athlete's device account, so it is something they choose.
AUTO_PUSH_SETTING = "hammerhead_auto_push_metrics"

router = APIRouter(
    prefix="/integrations/hammerhead",
    tags=["integrations"],
    dependencies=[pat_scopes(read="integrations:read", write="integrations:write")],
)


# ── Metrics push ───────────────────────────────────────────────────────────


class MetricsPushResponse(BaseModel):
    #: The Hammerhead field names that were sent. Anything absent was left
    #: unset here, and so stays editable by the athlete on the Karoo.
    pushed: list[str]


async def _connection(user_id: str, session: AsyncSession) -> ProviderConnection | None:
    return (
        await session.execute(
            select(ProviderConnection).where(
                ProviderConnection.user_id == user_id,
                ProviderConnection.provider == PROVIDER,
            )
        )
    ).scalar_one_or_none()


@router.post("/push-metrics", response_model=MetricsPushResponse,
             operation_id="pushHammerheadMetrics",
             summary="Push FTP, weight, heart rate and zones to the Karoo")
async def push_metrics(
    ctx_athlete=Depends(get_ctx_session_athlete),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    """Send the athlete's training metrics to their Hammerhead account.

    A partial update: only the metrics set here are sent, and the rest stay
    whatever the athlete set on the Karoo. Answers 403 ``insufficient_scope``
    when the athlete did not grant ``metrics:write``.
    """
    ctx, _session, athlete = ctx_athlete
    conn = await _connection(ctx.user_id, registry_session)
    if conn is None:
        raise HTTPException(status_code=400, detail="hammerhead is not connected")

    payload = build_metrics_payload(athlete)
    if not payload:
        raise HTTPException(status_code=422, detail="no_metrics_set")

    access_token = await ensure_fresh_token(conn, registry_session)
    try:
        await HammerheadClient().push_metrics(access_token, payload)
    except ScopeNotGranted:
        raise HTTPException(status_code=403, detail="insufficient_scope")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            raise HTTPException(status_code=403, detail="insufficient_scope")
        raise HTTPException(status_code=502, detail="Hammerhead API error during push")
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="Hammerhead API unreachable")
    return MetricsPushResponse(pushed=sorted(payload))


async def auto_push_metrics(user_id: str, athlete) -> None:
    """Push metrics in the background after a profile change, if opted in.

    Never raises: a profile save must not fail, or be slowed, because the Karoo
    was unreachable. The payload is built up front, so the athlete object need
    not outlive the request that changed it.
    """
    if not (athlete.app_settings or {}).get(AUTO_PUSH_SETTING):
        return
    payload = build_metrics_payload(athlete)
    if not payload:
        return

    async def _push() -> None:
        from backend.app.db.registry import _RegistrySessionLocal

        try:
            async with _RegistrySessionLocal() as reg_session:
                conn = await _connection(user_id, reg_session)
                if conn is None:
                    return
                access_token = await ensure_fresh_token(conn, reg_session)
            await HammerheadClient().push_metrics(access_token, payload)
            log.info("Pushed metrics to Hammerhead for user %s", user_id)
        except Exception:
            log.warning(
                "Automatic Hammerhead metrics push failed for user %s",
                user_id,
                exc_info=True,
            )

    asyncio.create_task(_push())


# ── Bridge poller (long-running background task) ───────────────────────────


def hammerhead_bridge_poller_configured() -> bool:
    """Whether this instance has a Hammerhead bridge to poll at all."""
    if not settings.hammerhead_bridge_url or not settings.hammerhead_bridge_secret:
        log.info("Hammerhead bridge not configured — poller inactive")
        return False
    log.info(
        "Hammerhead bridge poller armed (polling %s)", settings.hammerhead_bridge_url
    )
    return True


async def hammerhead_bridge_poller_once() -> None:
    """One poll. The loop and the leader claim live in ``backend.main``."""
    await _poll_bridge_once()


async def _poll_bridge_once() -> None:
    """Claim a batch, process it, ack what succeeded — the issue #50 protocol.

    Hammerhead never redelivers, so the bridge's queue is the only retry an
    event gets: a failure is nacked back to it rather than dropped.
    """
    async with httpx.AsyncClient(timeout=10.0) as http:
        client = BridgeClient(
            http, settings.hammerhead_bridge_url, settings.hammerhead_bridge_secret
        )
        events = await client.claim_batch()

        unprocessed = {e.get("id", ""): e.get("claim_token") for e in events}
        try:
            for event in events:
                event_id = event.get("id", "")
                claim_token = event.get("claim_token")
                try:
                    await process_hammerhead_webhook(event["payload"])
                except Exception:
                    log.exception("Failed to process Hammerhead bridge event %s", event_id)
                    await client.nack(event_id, claim_token)
                    unprocessed.pop(event_id, None)
                    continue

                await client.ack(event_id, claim_token)
                unprocessed.pop(event_id, None)
        except asyncio.CancelledError:
            for event_id, claim_token in unprocessed.items():
                with contextlib.suppress(Exception):
                    await client.nack(event_id, claim_token)
            raise
