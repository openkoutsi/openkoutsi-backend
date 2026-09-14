"""What the last backfill from each provider did, and whether it finished (issue #68).

``sync_provider_activities`` has four ways to end before it has walked an
athlete's history — the safety limit, a provider that stops serving detail data,
a 429, and a lost lease — and every one of them returned the same
``(count, earliest)`` tuple as a run that walked the lot. Nothing in the return
value, the API response or the database said the import was incomplete, so a
sync that imported 4 000 of 12 000 rides and one that imported all 4 000 there
were were the same event to everything downstream.

This module is the record that tells them apart. One
:class:`~backend.app.models.user_orm.ProviderSyncState` row per provider, in the
per-user database — so "one record per (user, provider)" is this file's primary
key — written at the start of a run and again when it ends.

Three things it is *for*, in the order they matter:

* **Telling a finished import from a stopped one.** ``status`` plus
  ``stop_reason``, read by the athlete through ``GET /integrations/status``.
* **Resuming.** ``resume_page`` is where the walk stopped, so the next run can
  step over history it has already walked instead of paying for it again.
* **Seeing a repeat as a repeat.** ``repeat_count`` counts runs that ended the
  same way in a row. Stopping in the same place three times is a provider
  problem or a poisoned range of activities, and no single run can see it.

Every function here commits. The state row is deliberately outside the caller's
transaction: it is written on the failure path, where the caller's own work is
being rolled back, and a record of the failure that rolls back with it would be
no record at all.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.user_orm import ProviderSyncState

log = logging.getLogger(__name__)


async def get_state(
    session: AsyncSession, provider: str
) -> ProviderSyncState | None:
    """The recorded state for one provider, or None if it has never run."""
    result = await session.execute(
        select(ProviderSyncState).where(ProviderSyncState.provider == provider)
    )
    return result.scalar_one_or_none()


async def _row(session: AsyncSession, provider: str) -> ProviderSyncState:
    state = await get_state(session, provider)
    if state is None:
        state = ProviderSyncState(
            provider=provider, status=ProviderSyncState.STATUS_RUNNING
        )
        session.add(state)
    return state


async def begin_run(session: AsyncSession, provider: str) -> ProviderSyncState:
    """Mark a run as started, and zero what this run has yet to report.

    What is *not* cleared is everything that describes accumulated knowledge
    rather than this run: ``stop_reason`` and ``repeat_count`` (the previous
    run's, until this one has its own answer), ``resume_page`` (which this run is
    about to act on), and ``oldest_seen_on`` (how far back the import has ever
    reached). A reader seeing ``running`` should read the stop fields as history;
    everything else on the row is about the run in progress.
    """
    state = await _row(session, provider)
    state.status = ProviderSyncState.STATUS_RUNNING
    state.started_at = datetime.now(timezone.utc)
    state.finished_at = None
    state.imported = 0
    state.listed = 0
    await session.commit()
    return state


async def record_completion(
    session: AsyncSession,
    provider: str,
    *,
    imported: int,
    listed: int,
    oldest_seen_on: date | None,
) -> ProviderSyncState:
    """Record a run that walked the provider's pages until they ran out.

    A completion is the only thing that clears ``resume_page``: there is nothing
    left to resume, and the next run walks from the front again — which is also
    what repairs any source an earlier run's fast-forward stepped over.
    """
    state = await _row(session, provider)
    state.status = ProviderSyncState.STATUS_COMPLETED
    state.stop_reason = None
    state.stop_detail = None
    state.resume_page = None
    state.repeat_count = 0
    state.repeat_since = None
    _finish(state, imported=imported, listed=listed, oldest_seen_on=oldest_seen_on)
    await session.commit()
    return state


async def record_stop(
    session: AsyncSession,
    provider: str,
    *,
    reason: str,
    detail: str | None,
    imported: int,
    listed: int,
    oldest_seen_on: date | None,
    resume_page: int | None,
) -> ProviderSyncState:
    """Record a run that ended before the history did.

    ``repeat_count`` is the reason this is a table rather than a log line: a run
    that stops the same way as the one before it increments the streak, and
    ``repeat_since`` keeps when the streak began. Three throttle stops in a row
    is a quota an athlete's history does not fit inside; three outage stops in a
    row is a provider problem or a poisoned range of activities. Neither is
    visible from one run.
    """
    state = await _row(session, provider)
    repeating = state.stop_reason == reason and state.repeat_count > 0
    state.status = ProviderSyncState.STATUS_STOPPED
    state.stop_reason = reason
    state.stop_detail = detail[:500] if detail else None
    state.repeat_count = state.repeat_count + 1 if repeating else 1
    if not repeating:
        state.repeat_since = datetime.now(timezone.utc)
    # **The cursor only ever advances; only a completion clears it.** A stop
    # shallower than what is recorded carries no new information about how far
    # the walk has got — a run that died before listing anything, or during the
    # front sweep every resumed run begins with — and letting it overwrite a
    # deep cursor throws away the saved walk this whole mechanism exists to
    # preserve. A front-sweep blip is the ordinary kind of failure, so without
    # this the cursor is lost routinely rather than rarely.
    #
    # It costs a *stale* cursor living longer across failed runs, which slightly
    # widens the fast-forward's mass-deletion window (see `_import_all_pages`).
    # That does not compound: only a run that settles the front and jumps can
    # act on a stale cursor, and such a run either completes — clearing it — or
    # stops deeper, replacing it. The runs that keep one alive are exactly the
    # ones that never jump.
    if resume_page is not None and (
        state.resume_page is None or resume_page > state.resume_page
    ):
        state.resume_page = max(1, resume_page)
    _finish(state, imported=imported, listed=listed, oldest_seen_on=oldest_seen_on)
    await session.commit()
    return state


def _finish(
    state: ProviderSyncState,
    *,
    imported: int,
    listed: int,
    oldest_seen_on: date | None,
) -> None:
    state.finished_at = datetime.now(timezone.utc)
    state.imported = imported
    state.listed = listed
    # Monotonic: the question it answers is "how far back has this import ever
    # reached", so a run that stopped on its first page must not erase the run
    # that reached 2012.
    if oldest_seen_on is not None and (
        state.oldest_seen_on is None or oldest_seen_on < state.oldest_seen_on
    ):
        state.oldest_seen_on = oldest_seen_on


async def resume_page_for(session: AsyncSession, provider: str) -> int | None:
    """The page a stopped run left for the next one, if there is one.

    Deliberately not conditioned on ``status``: the run asking this question has
    already marked the row ``running``, so a status check would only ever see its
    own footprint. The column itself carries the condition — only a stop sets it
    and only a completion clears it — so a non-NULL ``resume_page`` means the
    last run that ended did not finish.
    """
    state = await get_state(session, provider)
    return state.resume_page if state is not None else None


#: A provider that has never been synced. Not a null object — "no import has
#: run" is a state worth naming, and it is what a freshly connected account is in.
STATUS_NEVER = "never"

#: A row left saying ``running`` by a process that died mid-walk. The database
#: alone cannot tell this from a live import, which is why callers pass the
#: lease: a run that holds no lease is not running, whatever its row says. Named
#: rather than folded into ``stopped`` because the reason is different in kind —
#: nothing refused us, the server simply went away — and an athlete looking at a
#: spinner that will never finish deserves better than the spinner.
STATUS_INTERRUPTED = "interrupted"


def as_dict(state: ProviderSyncState | None, *, running: bool = False) -> dict:
    """The shape the API hands an athlete.

    ``running`` is whether a sync for this provider actually holds the lease
    right now. It is the caller's to supply because it is not in this row: the
    row says what the last run *claimed*, and a crashed run claims to be running
    forever.
    """
    if state is None:
        return {
            "status": STATUS_NEVER,
            "stop_reason": None,
            "stop_detail": None,
            "started_at": None,
            "finished_at": None,
            "imported": 0,
            "listed": 0,
            "oldest_seen_on": None,
            "more_expected": False,
            "repeat_count": 0,
            "repeat_since": None,
        }

    status = state.status
    if status == ProviderSyncState.STATUS_RUNNING and not running:
        status = STATUS_INTERRUPTED
    return {
        "status": status,
        # On a live run these describe the *previous* run — the row keeps them
        # until this one has an answer of its own.
        "stop_reason": state.stop_reason,
        "stop_detail": state.stop_detail,
        "started_at": state.started_at,
        "finished_at": state.finished_at,
        "imported": state.imported,
        "listed": state.listed,
        "oldest_seen_on": state.oldest_seen_on,
        # The one field a UI can act on without knowing the vocabulary: there is
        # history left that pressing Sync again would import.
        "more_expected": status
        in (ProviderSyncState.STATUS_STOPPED, STATUS_INTERRUPTED),
        "repeat_count": state.repeat_count,
        "repeat_since": state.repeat_since,
    }
