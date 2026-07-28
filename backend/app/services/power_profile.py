"""Rank-1 power bests and the CP/W' fits derived from them.

The "keep the single best power per duration, then fit a model to it" query is
needed in several places — the FTP estimate and power-model endpoints fit
against the athlete's *current* profile, while W' balance (issue #37) needs the
profile as it stood on a given activity's date. Both live here so there is one
implementation of the query rather than a copy per caller.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.user_orm import ActivityPowerBest
from openkoutsi.training_math import (
    CP_FIT_DURATIONS,
    cp_wprime_plausible,
    estimate_cp_wprime,
)


async def rank1_bests(
    athlete_id: str,
    session: AsyncSession,
    durations: list[int],
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[int, float]:
    """Best single effort per duration, as ``{duration_s: watts}``.

    ``since`` restricts to a rolling window (the ``?days=`` query parameters);
    ``until`` restricts to efforts on or before a point in time, which is how a
    historical activity gets the power profile the athlete actually had then
    instead of one that includes rides they hadn't done yet.

    Durations with no qualifying effort are omitted.
    """
    where = [
        ActivityPowerBest.athlete_id == athlete_id,
        ActivityPowerBest.duration_s.in_(durations),
    ]
    if since is not None:
        where.append(ActivityPowerBest.activity_start_time >= since)
    if until is not None:
        where.append(ActivityPowerBest.activity_start_time <= until)

    rows = await session.execute(
        select(ActivityPowerBest.duration_s, ActivityPowerBest.power_w)
        .where(*where)
        .order_by(ActivityPowerBest.duration_s, ActivityPowerBest.power_w.desc())
    )

    rank1: dict[int, float] = {}
    for duration_s, power_w in rows.all():
        if duration_s not in rank1:
            rank1[duration_s] = power_w
    return rank1


async def cp_wprime_as_of(
    athlete_id: str, session: AsyncSession, as_of: datetime | None
) -> tuple[float | None, float | None, int]:
    """Fit CP (watts) and W' (joules) from the bests available on ``as_of``.

    Applying today's all-time CP to a ride from two years ago would be
    anachronistic — the athlete simply wasn't that rider then — so the fit is
    restricted to efforts recorded on or before the date in question. Passing
    ``None`` fits the athlete's whole history.

    Returns ``(cp, w_prime, n_points)`` where ``n_points`` is how many duration
    bests the fit had to work with. Returns ``(None, None, n_points)`` when
    there aren't enough bests to fit **or when the fit is not physiologically
    plausible** — the OLS intercept is unconstrained, so a rider who only ever
    rides steady routinely fits a negative or near-zero W'. Rejecting here keeps
    "no CP → no columns, no stream" as the single failure mode, so the stored
    columns can never disagree with the presence of the stream.

    ``n_points`` is returned even on rejection: it is what makes a fit against a
    nearly-empty profile (a reverse-chronological backlog import) findable after
    the fact instead of indistinguishable from a well-supported one.
    """
    bests = await rank1_bests(
        athlete_id, session, CP_FIT_DURATIONS, until=as_of
    )
    cp, w_prime = estimate_cp_wprime(bests)
    if not cp_wprime_plausible(cp, w_prime):
        return None, None, len(bests)
    return cp, w_prime, len(bests)
