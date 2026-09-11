"""Reading current quota headroom out of the recorded rate-limit observations (issue #66).

The inversion worth stating plainly: **the provider tells us our usage
authoritatively in every response header.** Counting our own calls approximates a
number Strava already reports exactly. So this module — not the call counter — is
the answer to "can we start a big import right now", and the call counts in
``api_usage`` serve history, trend and attribution instead.

Because every response carries the headers, the reading lives on the
``api_usage`` row itself and "current headroom" is the newest row per service.
No snapshot table, no second write path, and a full history for trend charts for
free.

This is deliberately a *service function* rather than a query embedded in the
admin route: enforcement (throttling on low headroom) is issue #67's, and it
should be able to consume the same read model without a refactor.

Three things separate a useful panel from a misleading one, and all three live
here rather than in the caller:

* **Window rollover.** A usage reading is only valid inside the window it was
  observed in. Reporting last window's 580/600 against a window that is actually
  empty is worse than reporting nothing — it would stop an import that had full
  headroom.
* **Staleness.** Headroom is a *now* number, but it is only as fresh as our last
  call. The age of the observation travels with it so the UI can say so.
* **429s are ground truth.** The one unambiguous record that we exceeded, and
  the thing worth alerting on, so it is reported alongside.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.api_usage import api_usage_session_factory
from ..models.api_usage_orm import ApiUsage
from .api_usage import OUTCOME_RATE_LIMITED

#: Services whose responses we look for a quota reading on. Wahoo is listed even
#: though it publishes none: "this provider does not report a quota" is a
#: different, and more useful, answer than the service simply being absent.
QUOTA_SERVICES: tuple[str, ...] = ("strava", "wahoo")

#: Strava's short window. It is aligned to the quarter hour rather than being a
#: rolling 15 minutes from the observation — which is what makes rollover
#: computable from a timestamp alone.
SHORT_WINDOW = timedelta(minutes=15)


def _aware(dt: datetime) -> datetime:
    """SQLite hands back naive datetimes; every comparison here is in UTC."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def short_window_start(when: datetime) -> datetime:
    """Start of the quarter-hour window containing *when*."""
    when = _aware(when)
    return when.replace(minute=(when.minute // 15) * 15, second=0, microsecond=0)


def daily_window_start(when: datetime) -> datetime:
    """Start of the UTC day containing *when* — Strava's daily counter resets there."""
    when = _aware(when).astimezone(timezone.utc)
    return when.replace(hour=0, minute=0, second=0, microsecond=0)


@dataclass(frozen=True)
class WindowHeadroom:
    """One quota window's standing, as of ``observed_at``.

    ``usage`` is 0 with ``observed_in_window=False`` when the observation
    predates the current window: the counter has demonstrably reset, so 0 is the
    honest figure and the flag says it was inferred rather than seen. ``usage``
    is None only when we hold no reading at all for this window.
    """

    window: str        # short | daily
    scope: str         # overall | read
    usage: int | None
    limit: int | None
    window_start: datetime
    resets_at: datetime
    observed_in_window: bool

    @property
    def remaining(self) -> int | None:
        if self.usage is None or self.limit is None:
            return None
        return max(0, self.limit - self.usage)


@dataclass(frozen=True)
class ServiceHeadroom:
    """Everything the admin panel needs to render one service honestly."""

    service: str
    observed_at: datetime | None
    age_seconds: float | None
    windows: list[WindowHeadroom]
    last_rate_limited_at: datetime | None

    @property
    def has_observation(self) -> bool:
        return self.observed_at is not None


def _window(
    *,
    window: str,
    scope: str,
    usage: int | None,
    limit: int | None,
    observed_at: datetime | None,
    now: datetime,
) -> WindowHeadroom:
    if window == "short":
        start = short_window_start(now)
        resets_at = start + SHORT_WINDOW
    else:
        start = daily_window_start(now)
        resets_at = start + timedelta(days=1)

    in_window = observed_at is not None and _aware(observed_at) >= start
    if usage is not None and not in_window:
        # The window rolled over since we looked. The counter is empty; saying
        # otherwise would show alarming usage against a window nothing has spent.
        usage = 0
    return WindowHeadroom(
        window=window,
        scope=scope,
        usage=usage,
        limit=limit,
        window_start=start,
        resets_at=resets_at,
        observed_in_window=in_window,
    )


async def current_headroom(
    service: str,
    *,
    session: AsyncSession | None = None,
    now: datetime | None = None,
) -> ServiceHeadroom:
    """The latest quota standing for *service*.

    ``session`` is optional so a caller with one (the admin route) reuses it and
    a caller without (a throttle) needs no plumbing. ``now`` is injectable
    because window rollover is the one behaviour here that can only be tested by
    pinning the clock across a boundary.
    """
    if session is None:
        async with api_usage_session_factory()() as own:
            return await current_headroom(service, session=own, now=now)

    now = _aware(now or datetime.now(timezone.utc))

    # Newest row that actually carries a reading. Rows without one — every Wahoo
    # and email row, and any transport failure — must not shadow it.
    stmt = (
        select(ApiUsage)
        .where(ApiUsage.service == service)
        .where(
            or_(
                ApiUsage.ratelimit_limit_short.is_not(None),
                ApiUsage.ratelimit_limit_daily.is_not(None),
                ApiUsage.ratelimit_read_limit_short.is_not(None),
                ApiUsage.ratelimit_read_limit_daily.is_not(None),
            )
        )
        .order_by(ApiUsage.created_at.desc())
        .limit(1)
    )
    row = (await session.execute(stmt)).scalars().first()

    throttled_stmt = (
        select(ApiUsage.created_at)
        .where(ApiUsage.service == service)
        .where(ApiUsage.outcome == OUTCOME_RATE_LIMITED)
        .order_by(ApiUsage.created_at.desc())
        .limit(1)
    )
    last_429 = (await session.execute(throttled_stmt)).scalars().first()

    if row is None:
        return ServiceHeadroom(
            service=service,
            observed_at=None,
            age_seconds=None,
            windows=[],
            last_rate_limited_at=_aware(last_429) if last_429 else None,
        )

    observed_at = _aware(row.created_at)
    windows = [
        _window(window="short", scope="overall", usage=row.ratelimit_usage_short,
                limit=row.ratelimit_limit_short, observed_at=observed_at, now=now),
        _window(window="daily", scope="overall", usage=row.ratelimit_usage_daily,
                limit=row.ratelimit_limit_daily, observed_at=observed_at, now=now),
        _window(window="short", scope="read", usage=row.ratelimit_read_usage_short,
                limit=row.ratelimit_read_limit_short, observed_at=observed_at, now=now),
        _window(window="daily", scope="read", usage=row.ratelimit_read_usage_daily,
                limit=row.ratelimit_read_limit_daily, observed_at=observed_at, now=now),
    ]
    return ServiceHeadroom(
        service=service,
        observed_at=observed_at,
        age_seconds=max(0.0, (now - observed_at).total_seconds()),
        windows=[w for w in windows if w.limit is not None or w.usage is not None],
        last_rate_limited_at=_aware(last_429) if last_429 else None,
    )


async def all_headroom(
    *,
    session: AsyncSession | None = None,
    now: datetime | None = None,
    services: tuple[str, ...] = QUOTA_SERVICES,
) -> list[ServiceHeadroom]:
    """:func:`current_headroom` for every service we watch a quota on."""
    if session is None:
        async with api_usage_session_factory()() as own:
            return await all_headroom(session=own, now=now, services=services)
    return [await current_headroom(s, session=session, now=now) for s in services]
