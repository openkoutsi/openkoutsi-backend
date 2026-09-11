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

import asyncio
import contextlib
import logging
from contextvars import ContextVar
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Iterator, Optional

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

#: Whether the last write failed. Failures here repeat once per outbound HTTP
#: request, so a traceback each would turn "accounting is degraded" into "the
#: disk is full" — which is the contract this module exists to keep, broken one
#: indirection later. Reported in full once, then at debug until a write
#: succeeds, so a later unrelated failure is not permanently demoted.
_write_failing = False

#: In-flight recording tasks, held so the event loop cannot garbage-collect one
#: mid-write. ``asyncio`` keeps only a weak reference to a running task.
_pending: set[asyncio.Task] = set()


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


def schedule_api_usage(**fields: Any) -> None:
    """Record one call **off the caller's critical path**.

    This is the seam every hot caller should use. :func:`record_api_usage` can
    never *fail* a request — it swallows everything — but awaiting it inline can
    still *stall* one: the engine's pool and connect timeouts are measured in
    tens of seconds, so a database held under an exclusive lock (which is
    precisely what the ``VACUUM`` in the documented retention prune takes) turns
    every outbound provider request into a 30-second one. A backfill then does
    not fail, it crawls, and nothing in the admin panel explains why.

    ``created_at`` is stamped **here**, at the moment of the call, and carried
    into the write. SQLAlchemy evaluates a column default at INSERT time, so
    leaving it to the model would date each row by when its write happened to
    win the database — which, once the writes run concurrently, is not the order
    the calls were made in. Headroom reads the newest row for a service, so a
    later observation landing first would shadow the one that is actually
    current.

    Silently does nothing when there is no running loop — there is nothing to
    schedule onto, and a synchronous caller is not on a critical path worth
    protecting.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    fields.setdefault("created_at", datetime.now(timezone.utc))
    task = loop.create_task(record_api_usage(**fields))
    _pending.add(task)
    task.add_done_callback(_pending.discard)


async def drain_api_usage_writes() -> None:
    """Wait for every scheduled write to finish.

    Used on shutdown, so a redeploy does not drop the rows still in flight, and
    by tests, which would otherwise race the writes they assert on.
    """
    while _pending:
        await asyncio.gather(*list(_pending), return_exceptions=True)


async def record_api_usage(
    *,
    service: str,
    endpoint: str,
    method: str,
    status_code: int | None,
    duration_ms: int | None = None,
    user_id: str | None = None,
    rate_limit: RateLimitReading | None = None,
    created_at: datetime | None = None,
) -> None:
    """Record one outbound third-party call. Never raises into the caller.

    ``endpoint`` must already be a normalised template — see
    :func:`backend.app.services.providers.http.normalise_endpoint`. Nothing here
    inspects a URL, so nothing here can leak one.
    """
    global _write_failing
    reading = rate_limit or RateLimitReading()
    try:
        async with api_usage_session_factory()() as session:
            session.add(
                ApiUsage(
                    created_at=created_at or datetime.now(timezone.utc),
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
        _write_failing = False
    except Exception as exc:  # noqa: BLE001 - accounting must never break a sync
        if _write_failing:
            # Already reported. One line, no traceback: this path runs once per
            # outbound request, and a 10,000-request backfill against an
            # unwritable database would otherwise write ~90 MB of identical
            # tracebacks onto the same volume as DATA_DIR.
            log.debug(
                "Still failing to record API usage (service=%s endpoint=%s): %s",
                service,
                endpoint,
                exc,
            )
        else:
            _write_failing = True
            log.warning(
                "Failed to record API usage (service=%s endpoint=%s) — further "
                "failures will be logged at debug until one succeeds",
                service,
                endpoint,
                exc_info=True,
            )
