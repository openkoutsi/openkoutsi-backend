"""
Inbound-email background infrastructure (issue #38).

This module contains only the **inbound-email bridge poller** — a long-running
background task that polls the optional inbound-email bridge for received
messages and fans each one out to every administrator's in-app inbox via the
notifications service.

There is deliberately **no inbound HTTP endpoint on the backend**: the bridge is
a high-uptime public receiver that *holds* verified mail in a queue, and the
backend pulls from it on its own schedule (exactly like the Strava/Wahoo
bridges). That way mail is never lost while the backend is down — it simply
waits in the bridge until the next poll — and the backend never exposes a public
inbound surface.
"""

import asyncio
import logging

import httpx

from backend.app.core.config import settings
from backend.app.db.registry import get_registry_session
from backend.app.services import notifications

log = logging.getLogger(__name__)

# In-app messages carry only a short preview of the body; the full text stays in
# the provider's forwarded mailbox copy (if any). See the retention note in #38.
_SNIPPET_MAX_CHARS = 500


def _snippet(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= _SNIPPET_MAX_CHARS:
        return collapsed
    return collapsed[:_SNIPPET_MAX_CHARS].rstrip() + "…"


def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {settings.inbound_bridge_secret}"}


# ── Bridge poller (long-running background task) ───────────────────────────


async def inbound_bridge_poller() -> None:
    """
    Poll the inbound-email bridge every 60 seconds, deliver any pending messages
    to administrators' inboxes, and claim them so they aren't reprocessed.

    Silently no-ops unless the feature is enabled *and* the bridge URL/secret are
    configured — so an instance that doesn't run the optional bridge does nothing.
    """
    if not settings.inbound_email_enabled:
        log.info("Inbound email disabled — poller inactive")
        return
    if not settings.inbound_bridge_url or not settings.inbound_bridge_secret:
        log.info("Inbound bridge not configured — poller inactive")
        return

    log.info(
        "Inbound email poller started (polling %s)", settings.inbound_bridge_url
    )

    while True:
        await asyncio.sleep(60)
        try:
            await _poll_inbound_bridge_once()
        except Exception:
            log.exception("Inbound bridge poll failed")


async def _poll_inbound_bridge_once() -> None:
    async with httpx.AsyncClient(timeout=10.0) as http:
        try:
            r = await http.get(
                f"{settings.inbound_bridge_url}/events/pending",
                headers=_auth_headers(),
            )
            r.raise_for_status()
            events: list[dict] = r.json()
        except Exception:
            log.warning("Could not fetch events from inbound bridge")
            return

        if not events:
            return

        # One registry session for the whole batch (notify_admins only reads from
        # it; the per-user writes happen in each user's own DB).
        async for session in get_registry_session():
            for event in events:
                event_id = event.get("id", "")
                try:
                    await _deliver(session, event)
                except Exception:
                    log.exception(
                        "Failed to process inbound bridge event %s", event_id
                    )
                # Claim regardless of outcome (avoid infinite retry loops).
                try:
                    await http.post(
                        f"{settings.inbound_bridge_url}/events/{event_id}/claim",
                        headers=_auth_headers(),
                    )
                except Exception:
                    log.warning("Could not claim inbound bridge event %s", event_id)
            break


async def _deliver(session, event: dict) -> None:
    """Fan a single bridge event out to every administrator's inbox.

    When an operator address is configured, mail addressed elsewhere is dropped
    (the bridge/provider should only forward the operator address, but we filter
    defensively).
    """
    to_addr = event.get("to_addr") or event.get("to") or ""
    configured = settings.inbound_email_address.strip().lower()
    if configured and to_addr.strip().lower() != configured:
        log.info(
            "Inbound email for %r ignored (operator address is %r)",
            to_addr,
            settings.inbound_email_address,
        )
        return

    data = {
        "from": event.get("from_addr") or event.get("from") or "",
        "to": to_addr,
        "subject": event.get("subject") or "",
        "snippet": _snippet(event.get("text") or ""),
        "message_id": event.get("message_id") or "",
        "received_at": event.get("received_at"),
    }
    delivered = await notifications.notify_admins(
        session, notifications.INBOUND_EMAIL, data
    )
    log.info(
        "Inbound email from %r delivered to %d administrator(s)",
        data["from"],
        delivered,
    )
