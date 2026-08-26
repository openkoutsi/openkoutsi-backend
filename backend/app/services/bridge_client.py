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
import time
from typing import Optional

import httpx

log = logging.getLogger(__name__)

#: When to next probe a bridge that answered 404/405, keyed by base URL.
#:
#: A permanent latch would be wrong in exactly the deploy this fallback exists
#: for. Compose recreates the three images in whatever order it picks, so the
#: backend can be replaced first, meet the *old* bridge, latch — and then never
#: notice the bridge being replaced seconds later. It would run the pre-#50
#: flow, with every failure mode this change fixes silently live again, until
#: someone restarted it, which on a healthy deployment can be weeks.
#:
#: One extra request per bridge every five minutes against a 60-second poll,
#: and the window closes on its own.
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

        Returns events carrying a ``claim_token``; on a legacy bridge they carry
        ``None`` and the caller's ack/nack become no-ops, which reproduces the
        old behaviour exactly rather than half of it.
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
            response = await self._http.post(
                f"{self._base}{path}", headers=self._headers
            )
            # `httpx` raises only for transport failures, so without this a 401
            # (a secret rotated on one side) or a 500 (bridge disk full, the
            # column upgrade not applied) reads as success. This is the half of
            # the protocol that can wedge the queue, and it was the half with no
            # telemetry: a bridge that has stopped acking presents only as a
            # queue that mysteriously reprocesses everything every 15 minutes.
            response.raise_for_status()
        except httpx.HTTPError:
            # The visibility deadline still covers this: the event comes back
            # and the consumer's own idempotency check absorbs the replay. What
            # was missing was any record of *why*.
            log.warning(
                "Could not %s bridge event %s", verb, event_id, exc_info=True
            )
            return

        if response.json().get("status") == "stale":
            # A normal outcome — this claim lapsed and somebody else owns the
            # event. A spike in these is the signal that the visibility window
            # is too short for the drain.
            log.info("Bridge reported a stale %s for event %s", verb, event_id)
