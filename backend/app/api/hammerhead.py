"""
Hammerhead-specific background infrastructure (issue #69).

Connect / sync / disconnect go through the generic ``/integrations/*`` routes.
This module contains only the Hammerhead bridge poller — it claims queued
webhook events from ``hammerhead_bridge/`` and imports each one.
"""

import asyncio
import contextlib
import logging

import httpx
from fastapi import APIRouter

from backend.app.core.config import settings
from backend.app.services.bridge_client import BridgeClient
from backend.app.services.hammerhead_sync import process_hammerhead_webhook

log = logging.getLogger(__name__)

router = APIRouter(prefix="/hammerhead", tags=["hammerhead"])


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
