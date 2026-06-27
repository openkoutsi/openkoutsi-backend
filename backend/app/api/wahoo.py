"""
Wahoo-specific background infrastructure.

The direct webhook endpoint has been replaced by a bridge service (wahoo_bridge/).
This module contains only the Wahoo Bridge poller — a long-running background
task that polls the bridge service for webhook events and processes them.
"""

import asyncio
import logging

import httpx
from fastapi import APIRouter

from backend.app.core.config import settings
from backend.app.services.wahoo_sync import process_wahoo_webhook

log = logging.getLogger(__name__)

router = APIRouter(prefix="/wahoo", tags=["wahoo"])


# ── Bridge poller (long-running background task) ───────────────────────────

async def wahoo_bridge_poller() -> None:
    """
    Polls the Wahoo Bridge every 60 seconds, processes any pending webhook
    events, and claims them so they aren't reprocessed.

    Silently no-ops if WAHOO_BRIDGE_URL or WAHOO_BRIDGE_SECRET are not configured.
    """
    if not settings.wahoo_bridge_url or not settings.wahoo_bridge_secret:
        log.info("Wahoo bridge not configured — poller inactive")
        return

    log.info("Wahoo bridge poller started (polling %s)", settings.wahoo_bridge_url)

    while True:
        await asyncio.sleep(60)
        try:
            await _poll_bridge_once()
        except Exception:
            log.exception("Wahoo bridge poll failed")


async def _poll_bridge_once() -> None:
    async with httpx.AsyncClient(timeout=10.0) as http:
        try:
            r = await http.get(
                f"{settings.wahoo_bridge_url}/events/pending",
                headers={"Authorization": f"Bearer {settings.wahoo_bridge_secret}"},
            )
            r.raise_for_status()
            events: list[dict] = r.json()
        except Exception as e:
            log.warning(f"Could not fetch events from Wahoo bridge: {e}")
            return

        for event in events:
            event_id = event.get("id", "")

            try:
                await process_wahoo_webhook(event["payload"])
            except Exception:
                log.exception("Failed to process Wahoo bridge event %s", event_id)

            # Claim regardless of processing outcome (avoid infinite retry loops)
            try:
                await http.post(
                    f"{settings.wahoo_bridge_url}/events/{event_id}/claim",
                    headers={"Authorization": f"Bearer {settings.wahoo_bridge_secret}"},
                )
            except Exception:
                log.warning("Could not claim Wahoo bridge event %s", event_id)
