"""
Strava-specific background infrastructure.

OAuth connect/callback/sync/disconnect are now handled by the generic
/api/integrations/strava/* routes in integrations.py.

This module contains only the Strava Bridge poller — a long-running
background task that polls the bridge service for webhook events and
processes them.
"""

import asyncio
import logging

import httpx
from fastapi import APIRouter

from backend.app.core.config import settings
from backend.app.services.strava_sync import process_webhook_event

log = logging.getLogger(__name__)

router = APIRouter(prefix="/strava", tags=["strava"])


# ── Bridge poller (long-running background task) ───────────────────────────

async def strava_bridge_poller() -> None:
    """
    Polls the Strava Bridge every 60 seconds, processes any pending webhook
    events, and claims them so they aren't reprocessed.

    Silently no-ops if BRIDGE_URL or BRIDGE_SECRET are not configured.
    """
    if not settings.bridge_url or not settings.bridge_secret:
        log.info("Strava bridge not configured — poller inactive")
        return

    log.info("Strava bridge poller started (polling %s)", settings.bridge_url)

    while True:
        await asyncio.sleep(60)
        try:
            await _poll_bridge_once()
        except Exception:
            log.exception("Strava bridge poll failed")


async def _poll_bridge_once() -> None:
    async with httpx.AsyncClient(timeout=10.0) as http:
        # Fetch pending events
        try:
            r = await http.get(
                f"{settings.bridge_url}/events/pending",
                headers={"Authorization": f"Bearer {settings.bridge_secret}"},
            )
            r.raise_for_status()
            events: list[dict] = r.json()
        except Exception:
            log.warning("Could not fetch events from bridge")
            return

        for event in events:
            event_id = event.get("id", "")

            # process_webhook_event opens its own sessions internally
            try:
                await process_webhook_event(event)
            except Exception:
                log.exception("Failed to process bridge event %s", event_id)

            # Claim regardless of processing outcome (avoid infinite retry loops)
            try:
                await http.post(
                    f"{settings.bridge_url}/events/{event_id}/claim",
                    headers={"Authorization": f"Bearer {settings.bridge_secret}"},
                )
            except Exception:
                log.warning("Could not claim bridge event %s", event_id)
