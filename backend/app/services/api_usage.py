"""Recording outbound third-party API calls (issue #66).

The write path for ``api_usage.db``, and a deliberate copy of the one property
that matters most in :func:`~backend.app.services.llm_access.record_llm_usage`:
:func:`record_api_usage` opens its own short-lived session and **never raises
into the caller**. Usage accounting exists to tell us about a sync; it must
never be the thing that fails one.

Attribution is carried in a :class:`~contextvars.ContextVar` rather than
threaded through every provider method. The quota is per *application*, so the
user is diagnostic information ("whose backfill burned it"), not part of any
provider call's meaning — and a context variable keeps the ~17 call sites and
the provider ABC unchanged while still yielding NULL for the paths that
genuinely have no user (bridge polling, the OAuth exchange before an account is
linked).
"""

from __future__ import annotations

import contextlib
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional

from ..db.api_usage import api_usage_session_factory
from ..models.api_usage_orm import ApiUsage

log = logging.getLogger(__name__)

#: Outcome values for the ``outcome`` column. ``rate_limited`` is split out of
#: ``client_error`` on purpose: a 429 is the only unambiguous evidence that we
#: exceeded a quota, which makes it the one worth alerting on.
OUTCOME_OK = "ok"
OUTCOME_CLIENT_ERROR = "client_error"
OUTCOME_SERVER_ERROR = "server_error"
OUTCOME_TRANSPORT_ERROR = "transport_error"
OUTCOME_RATE_LIMITED = "rate_limited"

_user_id: ContextVar[Optional[str]] = ContextVar("api_usage_user_id", default=None)


@contextlib.contextmanager
def attribute_to_user(user_id: str | None) -> Iterator[None]:
    """Attribute every call made inside this block to *user_id*.

    Scoped to the context, so concurrent syncs for different athletes do not
    borrow each other's attribution.
    """
    token = _user_id.set(user_id)
    try:
        yield
    finally:
        _user_id.reset(token)


def current_user_id() -> str | None:
    return _user_id.get()


def outcome_for(status_code: int | None) -> str:
    """Classify a response status into the ``outcome`` column's vocabulary."""
    if status_code is None:
        return OUTCOME_TRANSPORT_ERROR
    if status_code == 429:
        return OUTCOME_RATE_LIMITED
    if status_code >= 500:
        return OUTCOME_SERVER_ERROR
    if status_code >= 400:
        return OUTCOME_CLIENT_ERROR
    return OUTCOME_OK


@dataclass(frozen=True)
class RateLimitReading:
    """One response's rate-limit headers, as numbers.

    ``short`` is the provider's 15-minute window and ``daily`` its 24-hour one.
    The ``read_*`` pair is Strava's separate, lower quota for read requests —
    kept apart rather than merged because a backfill is *all* reads, so the read
    ceiling is the one that binds first for exactly the question this feature
    exists to answer.
    """

    usage_short: int | None = None
    limit_short: int | None = None
    usage_daily: int | None = None
    limit_daily: int | None = None
    read_usage_short: int | None = None
    read_limit_short: int | None = None
    read_usage_daily: int | None = None
    read_limit_daily: int | None = None

    @property
    def is_empty(self) -> bool:
        return all(
            v is None
            for v in (
                self.usage_short,
                self.limit_short,
                self.usage_daily,
                self.limit_daily,
                self.read_usage_short,
                self.read_limit_short,
                self.read_usage_daily,
                self.read_limit_daily,
            )
        )


def _pair(raw: str | None) -> tuple[int | None, int | None]:
    """Split a ``"short,daily"`` header value into two ints.

    Returns ``(None, None)`` for anything unparseable — a header we cannot read
    is "no observation", never a zero, because a zero would read as "the window
    is empty" and invite exactly the wrong conclusion.
    """
    if not raw:
        return None, None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) < 2:
        return None, None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None, None


def parse_rate_limit(headers) -> RateLimitReading:
    """Read the rate-limit headers off a response.

    Strava sends ``X-RateLimit-Limit`` / ``X-RateLimit-Usage`` (the
    application-wide quota a 429 enforces) and ``X-ReadRateLimit-Limit`` /
    ``X-ReadRateLimit-Usage`` (the lower read-only quota), each a
    comma-separated *15-minute,daily* pair. Wahoo publishes no comparable quota;
    this simply finds nothing and the row's columns stay null.
    """
    limit_short, limit_daily = _pair(headers.get("X-RateLimit-Limit"))
    usage_short, usage_daily = _pair(headers.get("X-RateLimit-Usage"))
    read_limit_short, read_limit_daily = _pair(headers.get("X-ReadRateLimit-Limit"))
    read_usage_short, read_usage_daily = _pair(headers.get("X-ReadRateLimit-Usage"))
    return RateLimitReading(
        usage_short=usage_short,
        limit_short=limit_short,
        usage_daily=usage_daily,
        limit_daily=limit_daily,
        read_usage_short=read_usage_short,
        read_limit_short=read_limit_short,
        read_usage_daily=read_usage_daily,
        read_limit_daily=read_limit_daily,
    )


async def record_api_usage(
    *,
    service: str,
    endpoint: str,
    method: str,
    status_code: int | None,
    duration_ms: int | None = None,
    user_id: str | None = None,
    rate_limit: RateLimitReading | None = None,
) -> None:
    """Record one outbound third-party call. Never raises into the caller.

    ``endpoint`` must already be a normalised template — see
    :func:`backend.app.services.providers.http.normalise_endpoint`. Nothing here
    inspects a URL, so nothing here can leak one.
    """
    reading = rate_limit or RateLimitReading()
    try:
        async with api_usage_session_factory()() as session:
            session.add(
                ApiUsage(
                    service=service,
                    endpoint=endpoint,
                    method=method.upper(),
                    status_code=status_code,
                    outcome=outcome_for(status_code),
                    duration_ms=duration_ms,
                    user_id=user_id if user_id is not None else current_user_id(),
                    ratelimit_usage_short=reading.usage_short,
                    ratelimit_limit_short=reading.limit_short,
                    ratelimit_usage_daily=reading.usage_daily,
                    ratelimit_limit_daily=reading.limit_daily,
                    ratelimit_read_usage_short=reading.read_usage_short,
                    ratelimit_read_limit_short=reading.read_limit_short,
                    ratelimit_read_usage_daily=reading.read_usage_daily,
                    ratelimit_read_limit_daily=reading.read_limit_daily,
                )
            )
            await session.commit()
    except Exception:  # noqa: BLE001 - accounting must never break a sync
        log.warning(
            "Failed to record API usage (service=%s endpoint=%s)",
            service,
            endpoint,
            exc_info=True,
        )
