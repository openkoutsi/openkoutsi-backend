"""Garage orchestration: which bike a ride was done on, and how far it has gone.

Two jobs, both about ``Activity.bike_id``:

* **Automapping.** A bike claims a set of cycling ``sport_type`` values in
  ``Bike.default_sports``; a ride whose sport is claimed gets that bike. Run
  from every path that creates or reprocesses an activity.
* **Distance.** ``tracked_km`` is what openkoutsi actually observed —
  ``SUM(distance_m)`` over the rides assigned to a bike — and ``lifetime_km``
  adds the athlete's own ``odometer_base_km`` on top. Both derived on read, so
  reassigning a ride or fixing a baseline is immediately correct everywhere
  with no denormalised counter to drift.

**Applied, not suggested** — the opposite of :mod:`services.commute`, and
deliberately so. A bike assignment mints no achievements (no badge counts
bikes), removes nothing from any prompt (unlike a commute label, which hides a
ride from the RPE queue), and the feature's entire value *is* the total: a
per-bike distance counting only individually-confirmed rides is a chore, not a
garage. It is also trivially correctable in place, which is what the athlete
asked for.

The safety property #63 needed is still needed here; it is carried by
``Activity.bike_source`` instead of by a suggestion state. :func:`assign_bike`
writes only where that column is NULL or ``"auto"`` and never touches
``"manual"``. That is the invariant this module exists to hold.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from openkoutsi.sport_matching import CYCLING_SPORT_TYPES, canonical_sport_type
from backend.app.models.user_orm import Activity, Bike

log = logging.getLogger(__name__)

#: Written by :func:`assign_bike`, and the only value it will overwrite.
SOURCE_AUTO = "auto"

#: Written when the athlete picks a bike by hand. Never overwritten by any
#: automatic pass — not by a reprocess, a fresh provider sync, a history scan,
#: or an edit to what a bike claims.
SOURCE_MANUAL = "manual"

#: Ceiling on how many sports one bike may claim. The cycling vocabulary is
#: seven strings long, so this is a bound on nonsense rather than on anyone's
#: riding — but the column is JSON and reaches a per-athlete map built on every
#: ingest, so it does not go unbounded.
MAX_DEFAULT_SPORTS = len(CYCLING_SPORT_TYPES)

#: Rows per batch for the history scan. It writes, so it needs entities and
#: cannot select columns; streaming keeps peak memory flat across a history of
#: any size while each batch still flushes into the same transaction. Same
#: value and same reasoning as ``services.commute._YIELD_PER``.
_YIELD_PER = 1000


class SportClaimError(ValueError):
    """A ``default_sports`` list that cannot be stored as given.

    Carries the offending sport, and — for a collision — the bike already
    holding it, so the API can say *which* bike rather than just "no".
    """

    def __init__(self, message: str, *, sport: str, bike: Optional[Bike] = None):
        super().__init__(message)
        self.sport = sport
        self.bike = bike


def normalise_default_sports(raw: Optional[Iterable[str]]) -> Optional[list[str]]:
    """Canonical, de-duplicated cycling sport types, or None for "claims nothing".

    Input is normalised through :func:`canonical_sport_type`, so ``gravel_ride``
    and ``GravelRide`` land on the same key rather than becoming two claims that
    never both match. Anything outside the cycling set is rejected: a bike that
    claims ``Run`` would silently never match, which is a worse outcome than a
    422 the athlete can read.

    An empty list normalises to ``None`` — the column means "claims nothing"
    either way, and keeping one spelling of that keeps the API's answers stable.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        raise SportClaimError("default_sports must be a list", sport=raw)

    out: list[str] = []
    for value in raw:
        sport = canonical_sport_type(value) if isinstance(value, str) else None
        if sport is None or sport not in CYCLING_SPORT_TYPES:
            raise SportClaimError(
                f"Not a cycling sport type: {value!r}", sport=str(value)
            )
        if sport not in out:
            out.append(sport)
    if len(out) > MAX_DEFAULT_SPORTS:
        raise SportClaimError("Too many sports claimed", sport=out[MAX_DEFAULT_SPORTS])
    return out or None


def is_retired(bike: Bike) -> bool:
    return bike.retired_at is not None


async def check_sport_claims(
    session: AsyncSession,
    athlete,
    sports: Optional[list[str]],
    *,
    exclude_bike_id: Optional[str] = None,
) -> None:
    """Raise if another of the athlete's bikes already claims one of ``sports``.

    A sport may be claimed by **at most one bike per athlete**. Two bikes
    claiming ``GravelRide`` has no correct resolution — picking one is being
    wrong half the time — so the second claim is refused, naming the bike that
    already holds it.

    Retired bikes are not counted. Retiring the gravel bike and buying another
    is the ordinary case, and a claim held by a bike that no longer gets ridden
    would block the replacement for no benefit.
    """
    if not sports:
        return
    wanted = set(sports)
    result = await session.execute(
        select(Bike).where(Bike.athlete_id == athlete.id, Bike.retired_at.is_(None))
    )
    for other in result.scalars():
        if other.id == exclude_bike_id:
            continue
        clash = wanted.intersection(other.default_sports or ())
        if clash:
            sport = sorted(clash)[0]
            raise SportClaimError(
                f"{sport} is already claimed by bike {other.name!r}",
                sport=sport,
                bike=other,
            )


async def claim_map(session: AsyncSession, athlete) -> dict[str, str]:
    """``{sport_type: bike_id}`` for the athlete's **active** bikes.

    Retired bikes are excluded: a sold bike should not collect new rides. It
    keeps the rides it already has — see :func:`assign_bike`, which will not
    strip an assignment just because the bike it points at has been retired.
    """
    result = await session.execute(
        select(Bike.id, Bike.default_sports).where(
            Bike.athlete_id == athlete.id, Bike.retired_at.is_(None)
        )
    )
    mapping: dict[str, str] = {}
    for bike_id, sports in result:
        for sport in sports or ():
            # First claim wins if two somehow exist. `check_sport_claims` stops
            # that being created through the API; a database edited by hand, or
            # a bike un-retired into a collision, should still be deterministic
            # rather than dependent on row order.
            mapping.setdefault(sport, bike_id)
    return mapping


async def _retired_ids(session: AsyncSession, athlete) -> set[str]:
    result = await session.execute(
        select(Bike.id).where(
            Bike.athlete_id == athlete.id, Bike.retired_at.is_not(None)
        )
    )
    return set(result.scalars())


def assign(
    activity: Activity,
    claims: dict[str, str],
    *,
    retired_ids: frozenset[str] | set[str] = frozenset(),
) -> Optional[str]:
    """Point one activity at a bike. Returns the bike id written, or None.

    Does **not** commit — the caller owns the transaction, because every ingest
    path already has one open.

    Four outcomes, and each of the three that are not "assign it" is a case
    this feature goes quietly wrong without:

    - ``bike_source == "manual"`` — untouched, always. The athlete's correction
      has to survive a reprocess, a re-sync and an edit to what a bike claims.
    - the sport is not a cycling sport (or is unrecognised) — never assigned.
      A run does not belong to a bike, and guessing is worse than a blank.
    - the sport is claimed by an active bike — assigned, ``bike_source`` set to
      ``"auto"``.
    - the sport is claimed by nobody — left NULL if it was NULL. An existing
      ``"auto"`` assignment is withdrawn, so narrowing what a bike claims takes
      its rides back, *unless* the bike it points at is retired: a retired bike
      keeps its history, which is the entire difference between retiring and
      deleting.
    """
    if activity.bike_source == SOURCE_MANUAL:
        return None

    sport = canonical_sport_type(activity.sport_type)
    if sport is None or sport not in CYCLING_SPORT_TYPES:
        return None

    bike_id = claims.get(sport)
    if bike_id is None:
        if activity.bike_source == SOURCE_AUTO and activity.bike_id not in retired_ids:
            activity.bike_id = None
            activity.bike_source = None
        return None

    if activity.bike_id != bike_id or activity.bike_source != SOURCE_AUTO:
        activity.bike_id = bike_id
        activity.bike_source = SOURCE_AUTO
    return bike_id


async def assign_bike(
    session: AsyncSession, athlete, activity: Activity
) -> Optional[str]:
    """:func:`assign` for a single freshly-ingested or reprocessed activity.

    The hook every ingest path calls — provider sync, file processing, the
    manual-activity endpoint and reprocess. Missing any one of them produces the
    worst bug available here: a garage whose totals are right for rides that
    arrived one way and quietly short for rides that arrived another.

    Cheap when no bike claims anything, which is the common case, so it is safe
    to call unconditionally.
    """
    claims = await claim_map(session, athlete)
    if not claims and activity.bike_source != SOURCE_AUTO:
        return None
    return assign(activity, claims, retired_ids=await _retired_ids(session, athlete))


async def assign_history(session: AsyncSession, athlete) -> dict:
    """Look at the whole back catalogue and assign what the claims cover.

    The answer to a history that predates the garage: adding a bike, or changing
    what it claims, has to be able to look backwards, or a garage is empty
    exactly when it matters most.

    Only rows with ``bike_source IS NULL`` are touched. A ride already assigned
    — by hand or by an earlier automatic pass — is left exactly as it is, so
    this can never stomp a correction and never silently re-home a ride.

    An explicit request rather than something ``PATCH /api/bikes/{id}`` does
    inline: this walks the athlete's entire history, and doing that inside an
    edit holds a request worker for the length of the scan. Same precedent, and
    the same batching, as ``commute.scan_history``.
    """
    claims = await claim_map(session, athlete)
    if not claims:
        return {"scanned": 0, "assigned": 0}

    result = await session.stream_scalars(
        select(Activity)
        .where(Activity.athlete_id == athlete.id, Activity.bike_source.is_(None))
        .execution_options(yield_per=_YIELD_PER)
    )
    scanned = assigned = 0
    async for activity in result:
        scanned += 1
        if assign(activity, claims) is not None:
            assigned += 1
    if assigned:
        await session.commit()
    return {"scanned": scanned, "assigned": assigned}


async def tracked_km(session: AsyncSession, athlete) -> dict[str, float]:
    """``{bike_id: kilometres}`` openkoutsi has actually recorded per bike.

    Grouped in one query rather than one per bike — the garage lists every bike
    at once. Rides with ``distance_m IS NULL`` contribute nothing: SQL ``SUM``
    skips NULLs, and the ``coalesce`` covers the group where *every* row is
    NULL, which would otherwise come back as NULL and turn a real total into
    "unknown".
    """
    result = await session.execute(
        select(
            Activity.bike_id,
            func.coalesce(func.sum(Activity.distance_m), 0.0),
        )
        .where(Activity.athlete_id == athlete.id, Activity.bike_id.is_not(None))
        .group_by(Activity.bike_id)
    )
    return {bike_id: (total or 0.0) / 1000.0 for bike_id, total in result}


async def tracked_km_for(session: AsyncSession, athlete, bike_id: str) -> float:
    """:func:`tracked_km` for one bike, for the detail view."""
    result = await session.execute(
        select(func.coalesce(func.sum(Activity.distance_m), 0.0)).where(
            Activity.athlete_id == athlete.id, Activity.bike_id == bike_id
        )
    )
    return (result.scalar_one() or 0.0) / 1000.0


def lifetime_km(bike: Bike, tracked: float) -> float:
    """Everything the bike has ridden, including before openkoutsi saw it.

    Reported *alongside* ``tracked_km`` rather than instead of it: one is what
    openkoutsi observed and the other leans on a number the athlete typed, and a
    garage that blurs the two invites an argument it cannot win.
    """
    return (bike.odometer_base_km or 0.0) + tracked


def maintenance_order(entry) -> tuple:
    """Sort key for a maintenance log: by date, then by when it was recorded.

    ``performed_on`` is a date, so two things done on the same day tie. Falling
    back to ``created_at`` keeps the order the athlete entered them in, which is
    the only other information available about which came first — and component
    life is a difference between *consecutive* entries, so an unstable order
    would make the numbers move between reads.
    """
    return (entry.performed_on, entry.created_at, entry.id)


def component_spans(entries: list, lifetime: Optional[float]) -> dict[str, dict]:
    """Per-entry component life, keyed by entry id.

    Two numbers, and they answer different questions:

    - ``previous_component_km`` — how far the part replaced *at this entry* had
      run: the difference in ``odometer_km`` from the previous entry with the
      same ``component``. ``None`` when either reading is missing (the span is
      genuinely unknown, which is not the same as zero) or when nothing of that
      component came before.
    - ``km_since`` — how far the bike has run since this entry. On the newest
      entry for a component that is the open-ended case the athlete actually
      wants: the current tyres, fitted at 4 200 km, on a bike now at 6 000, have
      done 1 800.

    ``is_current`` marks that newest entry per component, so a client can show
    the running figure without re-deriving which row it belongs to.
    """
    ordered = sorted(entries, key=maintenance_order)
    previous: dict[str, object] = {}
    latest: dict[str, str] = {}
    out: dict[str, dict] = {}

    for entry in ordered:
        prior = previous.get(entry.component)
        span = None
        if (
            prior is not None
            and entry.odometer_km is not None
            and getattr(prior, "odometer_km", None) is not None
        ):
            span = entry.odometer_km - prior.odometer_km
        since = (
            lifetime - entry.odometer_km
            if lifetime is not None and entry.odometer_km is not None
            else None
        )
        out[entry.id] = {
            "previous_component_km": span,
            "km_since": since,
            "is_current": False,
        }
        previous[entry.component] = entry
        latest[entry.component] = entry.id

    for entry_id in latest.values():
        out[entry_id]["is_current"] = True
    return out
