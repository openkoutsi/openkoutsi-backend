"""Talking to a webhook bridge (issue #50).

Both bridges expose the same three-verb protocol and the backend consumes them
identically, so the client lives here once rather than twice in `api/`.

Claim-then-ack, not process-then-claim: the old order re-served an event whose
consumer died mid-import, and claiming first would have turned that duplicate
into a *loss* — the import is idempotent by `(provider, external_id)`, the LLM
analysis it triggers is not. So the claim carries a deadline, and an unacked
event becomes deliverable again when it expires.

The 404 fallback covers the deploy window where a new backend meets an old bridge
with no `/events/claim`. Deletable a release after both bridges have shipped.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

log = logging.getLogger(__name__)

#: When to next probe a bridge that answered 404/405, keyed by base URL.
#: Expires rather than latching: the backend can be recreated before the bridge,
#: latch against the old one, and otherwise never notice it being replaced
#: seconds later — running the pre-#50 flow until someone restarts it.
_LEGACY_UNTIL: dict[str, float] = {}
_RE_PROBE_AFTER_S = 300.0


class BridgeClient:
    """One bridge, one secret, one poll's worth of work."""

    def __init__(self, http: httpx.AsyncClient, base_url: str, secret: str):
        self._http = http
        self._base = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {secret}"}

    async def claim_batch(self) -> list[dict]:
        """Take a batch of events, each held until its claim expires.

        Events carry a ``claim_token``; on a legacy bridge it is ``None`` and
        ack/nack fall back to the old behaviour.
        """
        if _LEGACY_UNTIL.get(self._base, 0.0) <= time.monotonic():
            try:
                r = await self._http.post(
                    f"{self._base}/events/claim", headers=self._headers
                )
                if r.status_code not in (404, 405):
                    r.raise_for_status()
                    return r.json()
                log.warning(
                    "Bridge at %s has no /events/claim — falling back to the "
                    "pre-#50 fetch-then-claim flow until it is redeployed",
                    self._base,
                )
                _LEGACY_UNTIL[self._base] = time.monotonic() + _RE_PROBE_AFTER_S
            except httpx.HTTPError:
                log.warning("Could not claim events from bridge at %s", self._base)
                return []

        return await self._legacy_pending()

    async def _legacy_pending(self) -> list[dict]:
        try:
            r = await self._http.get(
                f"{self._base}/events/pending", headers=self._headers
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError:
            log.warning("Could not fetch events from bridge at %s", self._base)
            return []

    async def ack(self, event_id: str, claim_token: Optional[str]) -> None:
        """This event is done: never serve it again."""
        if claim_token is None:
            # Legacy bridge: the old `claim` verb is the only terminal one.
            await self._post(f"/events/{event_id}/claim", event_id, "claim")
            return
        await self._post(
            f"/events/{event_id}/ack?claim_token={claim_token}", event_id, "ack"
        )

    async def nack(self, event_id: str, claim_token: Optional[str]) -> None:
        """This attempt failed: return it to the queue after a backoff.

        A no-op on a legacy bridge, where an unclaimed event is already
        deliverable.
        """
        if claim_token is None:
            return
        await self._post(
            f"/events/{event_id}/nack?claim_token={claim_token}", event_id, "nack"
        )

    async def _post(self, path: str, event_id: str, verb: str) -> None:
        try:
            response = await self._http.post(
                f"{self._base}{path}", headers=self._headers
            )
            # `httpx` raises only for transport failures, so without this a 401
            # or a 500 reads as success — and a bridge that has stopped acking
            # would present only as a queue that reprocesses everything.
            response.raise_for_status()
        except httpx.HTTPError:
            # The visibility deadline covers this: the event comes back and the
            # consumer's idempotency check absorbs the replay.
            log.warning(
                "Could not %s bridge event %s", verb, event_id, exc_info=True
            )
            return

        if response.json().get("status") == "stale":
            # Normal: the claim lapsed and somebody else owns the event. A spike
            # means the visibility window is too short for the drain.
            log.info("Bridge reported a stale %s for event %s", verb, event_id)
