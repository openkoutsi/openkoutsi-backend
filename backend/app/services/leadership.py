"""One process at a time runs the background work (issue #50).

Without this, two processes each start their own copy of the three ``lifespan``
pollers, and both bridge pollers drain the same events — an activity imported,
analysed and billed twice.

The claim is taken per cycle rather than held as a term of office: a lease needs
a deadline to survive a dead holder, and no sane deadline spans the expiry
sweep's 24 hours. A process that loses a tick just skips it.

Renewal is still needed *within* a cycle (a bridge poll can be a hundred FIT
downloads), so the deadline is short and pushed out on its own task — a slow
drain cannot starve its own renewal.

Losing the lease mid-cycle cancels the work rather than finishing it: another
process may already be draining the same queue. Safe on all three — a bridge
event is not acked until processed, and the expiry sweep commits per token.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import AsyncIterator, Awaitable, Callable, Optional

from backend.app.db import leases
from backend.app.db.registry import registry_session
from backend.app.models.registry_orm import RegistryLease

log = logging.getLogger(__name__)

#: One lease for all three pollers — per-poller leases would let a process be
#: leader for Strava and not for Wahoo, for no benefit.
BACKGROUND_WORK = "background-work"

#: Four missed renewals, so a registry hiccup does not hand the work away but a
#: dead process frees it in two minutes.
LEASE_TTL = timedelta(seconds=120)
RENEW_EVERY_S = 30.0

#: Standby retry, jittered so simultaneous starts do not collide.
STANDBY_POLL_S = 15.0
STANDBY_JITTER_S = 5.0


async def _renew_until_lost(token: str, lost: asyncio.Event) -> None:
    """Keep the claim alive; set ``lost`` the moment it is not.

    Its own task, so renewal keeps its cadence however long the work takes. An
    exception counts as loss — working on a claim we could not confirm is the
    failure this module prevents, and being wrong costs one skipped tick.
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

    Yields an event set if the claim is ever lost, so the caller can stop.

    Uses ``acquire``/``renew``/``release`` rather than :func:`leases.hold`,
    whose warning on a failed acquisition would be a log line every 15 s
    forever on every standby. Failing to win an election is not a warning.
    """
    token: Optional[str] = None
    while token is None:
        async with registry_session() as session:
            # One attempt: a loser stands by rather than queueing behind the
            # winner for a lease measured in minutes.
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
        # Releasing here is what makes a redeploy hand over in milliseconds
        # rather than after a full TTL. Shielded because this runs on the
        # shutdown path, where the surrounding task is already cancelled.
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

    Returns when the claim is lost, mid-cycle if necessary. Callers run inside
    :func:`hold_background_work`, so "returned" means "stand by", not "done".
    """
    while not lost.is_set():
        cycle = asyncio.create_task(_guarded(work, label))
        loss = asyncio.create_task(lost.wait())
        done, _ = await asyncio.wait(
            {cycle, loss}, return_when=asyncio.FIRST_COMPLETED
        )
        if cycle not in done:
            # Claim lost first — stop here, another process may already be
            # doing the same work.
            cycle.cancel()
            try:
                await cycle
            except asyncio.CancelledError:
                pass
            log.info("Stopped %s mid-cycle after losing the lease", label)
        # Awaited, not just cancelled: an unretrieved cancelled task surfaces as
        # "Task was destroyed but it is pending" at loop teardown.
        loss.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loss
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
