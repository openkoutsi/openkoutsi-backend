"""Talking to a webhook bridge, with the claim semantics the queue now has.

Both bridges expose the same three-verb protocol, and the backend consumes both
identically, so the client lives here once rather than twice in `api/`.

**Why claim-then-ack rather than process-then-claim** (issue #50). The old order
processed an event and only then claimed it, so a consumer that died mid-import
left the event unclaimed and it was served again. Reversing that — claim first,
then process — would have converted the duplicate into a *loss*, which is worse
here: the import is idempotent by `(provider, external_id)` so a replay mostly
returns early, but the LLM analysis it triggers is not, and a lost one is
invisible.

So the claim carries a deadline. An event is handed out, and if no ack arrives
before it expires the event becomes deliverable again. At-least-once with a
bound, rather than at-most-once with a hole.

**The fallback matters more than it looks.** The backend and the two bridges are
three separate images, and the deploy recreates only the services whose digest
moved, in whatever order Compose picks. A new backend can therefore meet an old
bridge that has no `/events/claim`. Without the fallback below, that window is
one in which nothing drains either queue and the backlog grows against a ceiling
of 10 000. It is ten lines, and it can be deleted a release after both bridges
have shipped.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

log = logging.getLogger(__name__)

#: Set once per bridge URL when the new endpoint answers 404/405, so the probe
#: costs one request rather than one per poll.
_LEGACY_BRIDGES: set[str] = set()


class BridgeClient:
    """One bridge, one secret, one poll's worth of work."""

    def __init__(self, http: httpx.AsyncClient, base_url: str, secret: str):
        self._http = http
        self._base = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {secret}"}

    async def claim_batch(self) -> list[dict]:
        """Take a batch of events, each held until its claim expires.

        Returns events carrying a ``claim_token``; on a legacy bridge they carry
        ``None`` and the caller's ack/nack become no-ops, which reproduces the
        old behaviour exactly rather than half of it.
        """
        if self._base not in _LEGACY_BRIDGES:
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
                _LEGACY_BRIDGES.add(self._base)
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
        """This attempt failed: make it deliverable again now.

        On a legacy bridge there is nothing to release — an unclaimed event is
        already deliverable — so this is correctly a no-op there.
        """
        if claim_token is None:
            return
        await self._post(
            f"/events/{event_id}/nack?claim_token={claim_token}", event_id, "nack"
        )

    async def _post(self, path: str, event_id: str, verb: str) -> None:
        try:
            await self._http.post(f"{self._base}{path}", headers=self._headers)
        except httpx.HTTPError:
            # The visibility deadline covers a failed ack: the event comes back
            # and the consumer's own idempotency check absorbs the replay.
            log.warning("Could not %s bridge event %s", verb, event_id)
