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

Generic over the model so one implementation serves both databases — the
per-user DB (activity creation) and, in future, the registry (leader election for
the background pollers).
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

#: Default gap between attempts while waiting for a held lease.
POLL_SECONDS = 0.02


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
    deadline = asyncio.get_running_loop().time() + wait

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
        created = await session.execute(
            sqlite_insert(model)
            .values(name=name, holder=token, expires_at=now + ttl)
            .on_conflict_do_nothing(index_elements=[model.name])
        )
        await session.commit()
        if created.rowcount == 1:
            return token

        if asyncio.get_running_loop().time() >= deadline:
            return None
        await asyncio.sleep(poll)


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
    """
    token = await acquire(session, model, name, ttl=ttl, wait=wait)
    if token is None:
        log.warning("Gave up waiting for lease %s after %ss", name, wait)
    try:
        yield token
    finally:
        if token is not None:
            try:
                await release(session, model, name, token)
            except Exception:
                # The deadline covers this: the lease frees itself shortly.
                log.exception("Could not release lease %s", name)
