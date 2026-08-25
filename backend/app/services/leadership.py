"""One process at a time runs the background work (issue #50).

The three ``lifespan`` pollers — both bridge pollers and the token-expiry sweep
— are ``asyncio`` tasks started by whichever process boots. Two processes run
two of each, and for the bridge pollers that is not merely wasteful: each drains
the same unclaimed events, so an activity is imported twice, analysed twice and
billed twice. It is the reason ``DEPLOY.md`` says to run exactly one process,
and the reason ``--workers 2`` is not currently a safe thing to write down.

**A per-cycle claim, not a term of office.** The obvious shape is "elect a
leader, it holds leadership until it dies", and it is the wrong one here. Every
lease carries a deadline — that is how a holder that dies frees it — and the
token-expiry sweep's cycle is 24 hours, which no sane deadline can span. Holding
leadership across cycles therefore needs a renewal task purely to keep a claim
alive between two pieces of work, where claiming at the top of each cycle needs
nothing. A process that loses a tick simply skips it; the next one is 60 seconds
away.

Renewal is still needed *within* a cycle, because a cycle is not instant: a
bridge poll can process a hundred events, each a FIT download and parse. So the
deadline is short (a process that dies frees the work quickly) and pushed out
while the holder is demonstrably alive. The two concerns are separated onto
their own tasks, so a slow drain can never starve its own renewal.

**Losing the lease mid-cycle cancels the work.** Finishing the cycle first is
tempting and wrong: a lapsed deadline means another process may already be
draining the same queue, and two processes doing that is the duplicate import
this exists to prevent. Cancellation is safe on all three — a bridge event is
not claimed until it has been processed, so a cancelled one is simply re-served
on the next tick, and the expiry sweep commits its progress per token.
"""

from __future__ import annotations

import asyncio
import logging
import random
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import AsyncIterator, Awaitable, Callable, Optional

from backend.app.db import leases
from backend.app.db.registry import registry_session
from backend.app.models.registry_orm import RegistryLease

log = logging.getLogger(__name__)

#: The one lease name. "Who runs the background work" is a single decision —
#: one per poller would let a process be leader for Strava and not for Wahoo,
#: which is more states to reason about for no benefit.
BACKGROUND_WORK = "background-work"

#: How long a claim outlives the holder that stopped renewing it. Four missed
#: renewals, so a transient registry hiccup does not hand the work away, while a
#: process that actually died frees it inside two minutes rather than inside a
#: cycle.
LEASE_TTL = timedelta(seconds=120)
RENEW_EVERY_S = 30.0

#: How long a process that lost the claim waits before trying again. Jittered,
#: so N standbys that started together do not all wake into the same contention.
STANDBY_POLL_S = 15.0
STANDBY_JITTER_S = 5.0


async def _renew_until_lost(token: str, lost: asyncio.Event) -> None:
    """Keep the claim alive; set ``lost`` the moment it is not.

    Runs as its own task so that however long the work takes, renewal keeps its
    own cadence. An exception here is treated as loss: continuing to work on the
    strength of a claim we could not confirm is the failure mode this module
    exists to prevent, and the cost of being wrong is one skipped tick.
    """
    try:
        while True:
            await asyncio.sleep(RENEW_EVERY_S)
            async with registry_session() as session:
                still_ours = await leases.renew(
                    session, RegistryLease, BACKGROUND_WORK, token, ttl=LEASE_TTL
                )
            if not still_ours:
                log.warning(
                    "Lost the %s lease — another process holds it now", BACKGROUND_WORK
                )
                lost.set()
                return
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("Could not renew the %s lease — standing down", BACKGROUND_WORK)
        lost.set()


@asynccontextmanager
async def hold_background_work() -> AsyncIterator[asyncio.Event]:
    """Wait until this process owns the background work, then hold it.

    Yields an event that is set if the claim is ever lost, so the caller can
    stop rather than keep working on a claim it no longer has.

    Uses ``acquire``/``renew``/``release`` rather than :func:`leases.hold`
    deliberately: ``hold`` logs a warning every time it fails to take a lease,
    which for a standby polling every fifteen seconds would be a log line every
    fifteen seconds, forever, on every process that is not the leader. Failing
    to win an election is not a warning.
    """
    token: Optional[str] = None
    while token is None:
        async with registry_session() as session:
            # `wait=0` is exactly one attempt: a loser should go and stand by,
            # not queue behind the winner for a lease measured in minutes.
            token = await leases.acquire(
                session, RegistryLease, BACKGROUND_WORK, ttl=LEASE_TTL, wait=0
            )
        if token is None:
            await asyncio.sleep(
                STANDBY_POLL_S + random.uniform(0, STANDBY_JITTER_S)
            )

    log.info("Holding the %s lease — background pollers are live here", BACKGROUND_WORK)
    lost = asyncio.Event()
    renewer = asyncio.create_task(_renew_until_lost(token, lost))
    try:
        yield lost
    finally:
        renewer.cancel()
        try:
            await renewer
        except asyncio.CancelledError:
            pass
        # Releasing on the way out is what makes a redeploy hand over in
        # milliseconds instead of after a full TTL. Shielded, because this runs
        # on the shutdown path where the surrounding task is already being
        # cancelled, and an unreleased lease means no background work anywhere
        # until it expires.
        if not lost.is_set():
            try:
                await asyncio.shield(_release(token))
            except Exception:
                # The deadline covers this — it frees itself shortly.
                log.exception("Could not release the %s lease", BACKGROUND_WORK)


async def _release(token: str) -> None:
    async with registry_session() as session:
        await leases.release(session, RegistryLease, BACKGROUND_WORK, token)


async def run_until_lost(
    lost: asyncio.Event,
    work: Callable[[], Awaitable[None]],
    interval_s: float,
    *,
    label: str,
) -> None:
    """Run ``work`` every ``interval_s`` for as long as the claim holds.

    Returns when the claim is lost — mid-cycle if necessary. The caller is
    expected to be inside :func:`hold_background_work`, which is what makes
    "returned" mean "stand by and try again" rather than "finished".
    """
    while not lost.is_set():
        cycle = asyncio.create_task(_guarded(work, label))
        loss = asyncio.create_task(lost.wait())
        done, _ = await asyncio.wait(
            {cycle, loss}, return_when=asyncio.FIRST_COMPLETED
        )
        if cycle not in done:
            # The claim went first. Stop this cycle where it stands: another
            # process may already be doing the same work.
            cycle.cancel()
            try:
                await cycle
            except asyncio.CancelledError:
                pass
            log.info("Stopped %s mid-cycle after losing the lease", label)
        loss.cancel()
        if lost.is_set():
            return

        try:
            await asyncio.wait_for(lost.wait(), timeout=interval_s)
            return  # the claim lapsed while we were idle
        except asyncio.TimeoutError:
            pass  # the ordinary path: the interval elapsed, go again


async def _guarded(work: Callable[[], Awaitable[None]], label: str) -> None:
    """One cycle, with a failure logged rather than allowed to end the loop."""
    try:
        await work()
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("%s failed", label)
