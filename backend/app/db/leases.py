"""Named, expiring leases held in a database row (issue #50).

An ``asyncio.Lock`` coordinates tasks within one event loop and nothing else. For
the places where that is the *only* guard on a write, the guarantee is therefore
exactly as strong as "there is one process", which is an assumption the
deployment makes today and would like to stop making. A lease moves the decision
into the database, so it holds between processes as well as between tasks.

The mechanics are a conditional ``UPDATE``: a caller may take a lease only when
nobody holds it or the holder's deadline has passed, and the database — not the
caller — decides who won by whether the statement matched a row.

**Every lease expires.** A holder that dies mid-section cannot release, so
without a deadline the first crash would wedge that lease forever. The deadline
is therefore a crash-recovery bound rather than a timeout, and it must be
comfortably longer than the section it protects: expiring under a holder that is
merely slow hands the same lease to two callers, which is the failure the lease
exists to prevent.

Generic over the *model*, so one implementation serves both databases — the
per-user DB (activity creation) and, in future, the registry (leader election for
the background pollers). Not generic over the *dialect*: the claim-on-first-use
insert is ``sqlalchemy.dialects.sqlite``'s ``ON CONFLICT DO NOTHING``. Every
database in this application is SQLite, so that costs nothing today; it is
recorded here so the reuse above is not mistaken for portability.
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
    time. A caller that gets ``None`` still runs: this is a guard on a race, not
    an admission gate, and refusing to import an activity because a lease was
    busy would turn a rare duplicate into a routine failure. The warning is what
    makes the degraded case visible.

    **This owns the session's transaction boundaries.** Taking and releasing a
    lease both have to be visible to other connections, so both commit — and
    ``commit`` flushes first. A block that raises therefore gets an explicit
    rollback before the release, because otherwise the act of tidying up after a
    failure would publish the very work the failure should have discarded: on the
    attach path in ``provider_sync`` that is a flushed ``ActivitySource`` and the
    deletion of every stream and best belonging to the activity. Do not carry
    unrelated uncommitted work across this block.
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
