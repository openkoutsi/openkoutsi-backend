"""Telling "the provider has nothing" apart from "the provider is refusing us".

Every provider fetch in the sync pipeline used to be wrapped in a bare
``except Exception``, which made those two answers the same one. They are not:
absence is final and a throttle is a *later*, and recording a throttle as
absence is what imported activities with no power, HR or cadence behind them
and then never looked at them again (issue #67).

This module is the classifier. :func:`as_throttle` recognises the statuses that
mean "ask again later" and reads ``Retry-After`` off the response;
:func:`call_with_retry_after` waits out a throttle that told us how long, and
raises :class:`ProviderThrottled` when it did not or when the wait would be
longer than a sync should hold itself open for.

It deliberately does **not** pace outbound calls. A token bucket sized to the
provider's real quota needs the headroom reading that issue #66 adds, and
parsing those headers here would only have to be undone when it lands.
"""

from __future__ import annotations

import asyncio
import email.utils
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import httpx

log = logging.getLogger(__name__)

#: The throttle proper: a statement about our standing with the provider, not
#: about the thing we asked for. Nothing else we ask will fare better.
RATE_LIMIT_STATUSES = frozenset({429})

#: The provider having a bad moment. Also not an answer about this activity's
#: data — but, unlike a throttle, it says nothing about the *next* activity, and
#: treating the two the same let one permanently-500 ride stop the walk at the
#: same place on every sync, walling off everything older than it.
SERVER_ERROR_STATUSES = frozenset({500, 502, 503, 504})

#: Statuses that mean "ask again later" rather than "there is nothing here".
RETRYABLE_STATUSES = RATE_LIMIT_STATUSES | SERVER_ERROR_STATUSES

#: Longest ``Retry-After`` we will hold a sync open for. Strava's quota windows
#: are 15 minutes, and blocking a background task that long to save one request
#: is worse than stopping and resuming on the next sync — which, since the
#: import is now repairable, costs nothing but time.
MAX_RETRY_AFTER_WAIT = 60.0

#: One try, then one more after honouring ``Retry-After``. A second wait would
#: be waiting on a provider that has already told us once it is not ready.
_ATTEMPTS = 2


class ProviderThrottled(Exception):
    """The provider is rate-limiting or failing us — this is not an answer.

    Raised in place of the swallowed exception so callers can tell a throttle
    from an activity that genuinely has no data, and stop rather than import a
    hollow one.
    """

    def __init__(
        self,
        provider: str,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.provider = provider
        self.status_code = status_code
        #: Seconds the provider asked us to wait, when it said so.
        self.retry_after = retry_after
        detail = f"HTTP {status_code}" if status_code is not None else "no status"
        wait = f", retry after {retry_after:.0f}s" if retry_after is not None else ""
        super().__init__(f"{provider} is refusing us ({detail}{wait})")

    @property
    def is_rate_limit(self) -> bool:
        """Whether this is about our standing with the provider, or about one request.

        A 429 says every further request will meet the same wall, so a walk that
        hits one should stop. A 5xx says this request failed; the next activity
        is a fresh question, and stopping on it makes one unservable ride a
        permanent barrier to everything the walk had not yet reached.
        """
        return self.status_code in RATE_LIMIT_STATUSES


def parse_retry_after(value: str | None) -> float | None:
    """Seconds to wait, from a ``Retry-After`` header.

    RFC 9110 allows either a delta in seconds or an HTTP-date, and providers use
    both. Returns None when the header is absent or unparseable — which the
    callers treat as "it did not say", not as "wait zero".
    """
    if not value:
        return None
    raw = value.strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        # The only failure mode since 3.10: it raises rather than returning None.
        return None
    if when.tzinfo is None:
        # A zoneless HTTP-date. UTC is the only reading of one that is ever
        # right, and it has to be filled in before the subtraction below —
        # naive minus aware raises.
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def as_throttle(exc: BaseException, provider: str) -> ProviderThrottled | None:
    """The throttle this exception represents, or None if it is a real answer.

    Only ``httpx.HTTPStatusError`` carries a status, so anything else — a
    transport error, a parse failure, a provider client raising its own — is
    left to the caller's own judgement.
    """
    if isinstance(exc, ProviderThrottled):
        return exc
    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    response = getattr(exc, "response", None)
    if response is None or response.status_code not in RETRYABLE_STATUSES:
        return None
    return ProviderThrottled(
        provider,
        status_code=response.status_code,
        retry_after=parse_retry_after(response.headers.get("Retry-After")),
    )


def from_response(response, provider: str) -> ProviderThrottled | None:
    """The throttle this *response* represents, for clients that never raise.

    Some provider clients inspect the status themselves and return ``None``
    rather than calling ``raise_for_status`` — which is how a 429 reached the
    sync pipeline wearing the same face as "this workout has no FIT" (issue
    #67). They classify the response with this instead.
    """
    if response is None or response.status_code not in RETRYABLE_STATUSES:
        return None
    return ProviderThrottled(
        provider,
        status_code=response.status_code,
        retry_after=parse_retry_after(response.headers.get("Retry-After")),
    )


def is_definitive(exc: BaseException) -> bool:
    """Whether this failure is the provider's final word on the request.

    A 404 for an activity's streams is an answer: that activity has none, and
    asking again next sync will get the same 404 forever. A timeout is not.
    Only the first kind may settle a source as fetched.
    """
    if isinstance(exc, ProviderThrottled):
        return False
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    response = getattr(exc, "response", None)
    return response is not None and response.status_code not in RETRYABLE_STATUSES


async def call_with_retry_after(
    call: Callable[..., Awaitable[Any]],
    *args: Any,
    provider: str,
    what: str,
    may_wait: bool = True,
) -> Any:
    """Make one provider call, honouring ``Retry-After`` once.

    Raises :class:`ProviderThrottled` when the provider is still refusing us —
    either because it named a wait longer than :data:`MAX_RETRY_AFTER_WAIT`, or
    because it named none at all, or because it threw us out again after we
    waited. Every other exception propagates untouched, so a caller that wants
    to treat a plain failure as absence still can.

    ``may_wait=False`` for a call made while a lease is held. Sleeping there
    spends someone else's deadline: the activity-create lease covers a section
    already sized for a FIT download and parse, and two minutes of politeness on
    top of that is how a merely slow holder becomes an expired one, handing the
    same lease to two callers.
    """
    attempts = 0
    while True:
        try:
            return await call(*args)
        except Exception as exc:
            throttle = as_throttle(exc, provider)
            if throttle is None:
                raise
            attempts += 1
            delay = throttle.retry_after
            if (
                not may_wait
                or attempts >= _ATTEMPTS
                or delay is None
                or delay > MAX_RETRY_AFTER_WAIT
            ):
                raise throttle from exc
            log.warning(
                "%s throttled the %s request (HTTP %s) — waiting %.0fs as asked",
                provider,
                what,
                throttle.status_code,
                delay,
            )
            await asyncio.sleep(delay)
