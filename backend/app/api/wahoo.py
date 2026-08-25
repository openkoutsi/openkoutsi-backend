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
from backend.app.services.bridge_client import BridgeClient
from backend.app.services.wahoo_sync import process_wahoo_webhook

log = logging.getLogger(__name__)

router = APIRouter(prefix="/wahoo", tags=["wahoo"])


# ── Bridge poller (long-running background task) ───────────────────────────

def wahoo_bridge_poller_configured() -> bool:
    """Whether this instance has a Wahoo bridge to poll at all.

    Checked before contending for the background-work lease, so an instance with
    no Wahoo bridge never takes leadership it has nothing to do with.
    """
    if not settings.wahoo_bridge_url or not settings.wahoo_bridge_secret:
        log.info("Wahoo bridge not configured — poller inactive")
        return False
    log.info("Wahoo bridge poller armed (polling %s)", settings.wahoo_bridge_url)
    return True


async def wahoo_bridge_poller_once() -> None:
    """One poll. The loop and the leader claim live in ``backend.main``."""
    await _poll_bridge_once()


async def _poll_bridge_once() -> None:
    """Claim a batch, process it, ack what succeeded (issue #50).

    The claim carries a deadline, so an event whose consumer dies mid-import
    becomes deliverable again rather than being silently retired. That is why
    the ack is conditional on success where the old `claim` call was not: with
    `attempts` bounding redelivery, a transient failure no longer has to be
    treated as terminal to avoid a retry loop.
    """
    async with httpx.AsyncClient(timeout=10.0) as http:
        client = BridgeClient(http, settings.wahoo_bridge_url, settings.wahoo_bridge_secret)
        events = await client.claim_batch()

        for event in events:
            event_id = event.get("id", "")
            claim_token = event.get("claim_token")

            # `process_wahoo_webhook` opens its own sessions internally.
            try:
                await process_wahoo_webhook(event["payload"])
            except Exception:
                log.exception("Failed to process bridge event %s", event_id)
                # Deliverable again immediately, rather than at the deadline.
                # A genuinely poisonous event still retires once it has used up
                # `max_delivery_attempts` on the bridge.
                await client.nack(event_id, claim_token)
                continue

            await client.ack(event_id, claim_token)