"""Named, expiring leases held in a database row (issue #50).

An ``asyncio.Lock`` coordinates tasks within one event loop and nothing else, so
where it is the *only* guard on a write the guarantee is exactly as strong as
"there is one process". A lease moves the decision into the database, so it holds
between processes as well as between tasks.

The mechanics are a conditional ``UPDATE``: a caller may take a lease only when
nobody holds it or the holder's deadline has passed, and the database — not the
caller — decides who won, by whether the statement matched a row.

**Every lease expires.** A holder that dies mid-section cannot release, so
without a deadline the first crash would wedge the lease forever. It is a
crash-recovery bound rather than a timeout, and must be comfortably longer than
the section it protects: expiring under a merely slow holder hands the same lease
to two callers.

Generic over the *model*, so one implementation serves the per-user DB (activity
creation) and the registry (leader election). Not generic over the *dialect* —
the claim-on-first-use insert is ``sqlalchemy.dialects.sqlite``'s ``ON CONFLICT
DO NOTHING`` — so the reuse above is not portability.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Optional, Type

from sqlalchemy import DateTime, String, or_, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

log = logging.getLogger(__name__)

#: Gap before the first re-attempt when the lease is held. Each further attempt
#: doubles it, up to :data:`MAX_POLL_SECONDS`.
#:
#: The backoff is not politeness. A poll is a write transaction, SQLite has one
#: write lock per database, and the thing the waiter is waiting for is the holder
#: *committing* — so polling hard actively delays the event being polled for. The
#: sections under a lease are measured in seconds (a FIT download and parse, on
#: the attach path), which a half-second poll tracks perfectly well.
POLL_SECONDS = 0.02
MAX_POLL_SECONDS = 0.5


class LeaseMixin:
    """Columns every lease table carries.

    ``holder`` is the token the acquirer generated, not an identity anyone can
    guess: release is conditional on it, so a caller whose lease expired and was
    taken by someone else cannot release a lease it no longer owns.
    """

    name: Mapped[str] = mapped_column(String, primary_key=True)
    holder: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


async def acquire(
    session: AsyncSession,
    model: Type[LeaseMixin],
    name: str,
    *,
    ttl: timedelta,
    wait: float,
    poll: float = POLL_SECONDS,
) -> Optional[str]:
    """Take the lease named ``name``, waiting up to ``wait`` seconds.

    Returns the holder token, or ``None`` if the wait ran out. Commits — the
    claim has to be visible to other connections to mean anything, which is the
    same reason the activity insert it guards has to be committed rather than
    merely flushed.
    """
    token = uuid.uuid4().hex
    loop = asyncio.get_running_loop()
    deadline = loop.time() + wait
    # The insert can only ever match while the row is absent, which is decided
    # once and for all on the first pass. Re-trying it every poll was doubling
    # the wait loop's write traffic to no possible effect.
    insert_attempted = False
    delay = poll

    while True:
        now = datetime.now(timezone.utc)

        # The steady-state path: the row exists and is free (or abandoned).
        taken = await session.execute(
            update(model)
            .where(
                model.name == name,
                or_(
                    model.holder.is_(None),
                    model.expires_at.is_(None),
                    model.expires_at <= now,
                ),
            )
            .values(holder=token, expires_at=now + ttl)
            # SQLite hands these columns back naive, which the in-Python
            # evaluator cannot compare against an aware `now`; and the database
            # is the authority on who won regardless.
            .execution_options(synchronize_session=False)
        )
        await session.commit()
        if taken.rowcount == 1:
            return token

        # First ever use of this lease name — insert it already claimed, so the
        # create and the claim are one statement and two callers racing to
        # create it cannot both come away holding it.
        if not insert_attempted:
            insert_attempted = True
            created = await session.execute(
                sqlite_insert(model)
                .values(name=name, holder=token, expires_at=now + ttl)
                .on_conflict_do_nothing(index_elements=[model.name])
            )
            await session.commit()
            if created.rowcount == 1:
                return token

        remaining = deadline - loop.time()
        if remaining <= 0:
            return None
        await asyncio.sleep(min(delay, remaining))
        delay = min(delay * 2, MAX_POLL_SECONDS)


async def release(
    session: AsyncSession, model: Type[LeaseMixin], name: str, token: str
) -> None:
    """Give up a lease. Conditional on still holding it."""
    await session.execute(
        update(model)
        .where(model.name == name, model.holder == token)
        .values(holder=None, expires_at=None)
        .execution_options(synchronize_session=False)
    )
    await session.commit()


async def renew(
    session: AsyncSession,
    model: Type[LeaseMixin],
    name: str,
    token: str,
    *,
    ttl: timedelta,
) -> bool:
    """Push a held lease's deadline out. Returns whether we still held it.

    :func:`acquire` cannot do this: it mints a fresh token and only matches a row
    that is free or expired, so a holder calling it would fail or come back with
    a *different* token. Renewal is the same conditional ``UPDATE`` pointed at
    the row we already own.

    **A ``False`` return is the loss signal and is authoritative** — the deadline
    lapsed and somebody else took the lease, or the row is gone, and a caller
    working through it is the second writer the lease exists to prevent.

    This is what lets a lease outlive its section without a long deadline: a dead
    holder must free it within seconds, which is incompatible with a cycle
    measured in hours unless the deadline is pushed out while the holder is
    demonstrably alive.
    """
    now = datetime.now(timezone.utc)
    result = await session.execute(
        update(model)
        .where(model.name == name, model.holder == token)
        .values(expires_at=now + ttl)
        .execution_options(synchronize_session=False)
    )
    await session.commit()
    return result.rowcount == 1


@asynccontextmanager
async def hold(
    session: AsyncSession,
    model: Type[LeaseMixin],
    name: str,
    *,
    ttl: timedelta,
    wait: float,
) -> AsyncIterator[Optional[str]]:
    """Hold a lease for the duration of the block.

    Yields the holder token, or ``None`` when the lease could not be taken in
    time. A caller that gets ``None`` still runs: this guards a race, it is not an
    admission gate, and refusing an import because a lease was busy would turn a
    rare duplicate into a routine failure. The warning makes that visible.

    **This owns the session's transaction boundaries.** Taking and releasing must
    both be visible to other connections, so both commit — and ``commit`` flushes
    first. A block that raises gets an explicit rollback before the release, or
    tidying up after a failure would publish the work the failure should have
    discarded. Do not carry unrelated uncommitted work across this block.
    """
    token = await acquire(session, model, name, ttl=ttl, wait=wait)
    if token is None:
        log.warning(
            "Gave up waiting for lease %s after %ss — proceeding with in-process "
            "locking only, which is this process's own tasks and nothing else",
            name,
            wait,
        )

    failed = False
    try:
        yield token
    except BaseException:
        failed = True
        raise
    finally:
        if token is not None:
            try:
                if failed and session.in_transaction():
                    # Discard the block's half-finished work *before* the release
                    # commits. This also un-poisons a session left in
                    # pending-rollback by a failed flush, which would otherwise
                    # make the release below raise and strand the lease for its
                    # entire TTL.
                    await session.rollback()
                await release(session, model, name, token)
            except Exception:
                # The deadline covers this: the lease frees itself shortly.
                log.exception("Could not release lease %s", name)
