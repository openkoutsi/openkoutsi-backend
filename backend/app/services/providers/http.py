"""The one place every outbound provider HTTP request passes through (issue #66).

There is no shared HTTP client in this package — :mod:`.base` is a pure ABC with
no transport, and each provider method opens its own ``httpx.AsyncClient``. This
module supplies the client those call sites build instead, so that every request
is counted exactly once.

**Counting happens at the HTTP-request layer, not the provider-method layer**,
because quotas are counted in requests and one method call is not one request:
``StravaProviderClient.fetch_zones()`` issues two (``/athlete`` and
``/athlete/zones``, gathered), and ``WahooClient.download_fit_file()`` may follow
a CDN redirect. Counting per method would under-report precisely where the
backfill spends the most.

A transport wrapper is used rather than ``event_hooks`` for three reasons, the
last decisive:

* a response hook never runs for a **transport failure** — DNS, TLS and connect
  errors produce no response at all, so those requests would vanish;
* redirects and connection-level retries pass through the transport
  individually, which is what makes the count match the provider's;
* it is the single point that sees **every response's headers**, which is where
  the rate-limit reading comes from.
"""

from __future__ import annotations

import re
import time
from typing import Optional

import httpx

from backend.app.services.api_usage import parse_rate_limit, record_api_usage

#: Hosts we own the URL shape of, and the path prefix to strip from each so the
#: recorded template reads like the provider's documented endpoint rather than
#: its version scheme. Anything not listed is a redirect target (a CDN), whose
#: path is a signed blob and is never recorded — see :func:`normalise_endpoint`.
_KNOWN_HOSTS: dict[str, tuple[str, ...]] = {
    "www.strava.com": ("/api/v3",),
    "api.wahooligan.com": ("/v1",),
}

_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                      r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]{8,}$")


def _is_identifier(segment: str) -> bool:
    """Whether a path segment is an id rather than part of the route's name.

    Deliberately generous: a segment that is *probably* an identifier is
    replaced, because the cost of a false positive is a slightly coarser bucket
    in an admin table, while the cost of a false negative is an athlete's
    activity id in a database that exists to count things.
    """
    if not segment:
        return False
    if segment.isdigit():
        return True
    if _UUID_RE.match(segment) or _HEX_RE.match(segment):
        return True
    # An opaque token: long, and mixing digits in with the letters. Real route
    # names in these APIs are short words ("activities", "power_zones").
    return len(segment) >= 16 and any(c.isdigit() for c in segment)


def normalise_endpoint(url: httpx.URL) -> str:
    """A storable template for *url* — never the URL itself.

    ``/api/v3/activities/12345/streams`` becomes ``/activities/{id}/streams``.
    Raw URLs carry activity ids and, on some paths, query-string credentials, so
    the query is dropped whole and every segment that looks like an identifier is
    replaced. A host we do not recognise is a redirect target whose path is a
    signed blob: it is recorded as ``<host>/*`` — enough to attribute the request
    and to group CDN downloads together, and nothing more.
    """
    host = url.host
    prefixes = _KNOWN_HOSTS.get(host)
    if prefixes is None:
        return f"{host}/*"

    path = url.path or "/"
    for prefix in prefixes:
        if path.startswith(prefix):
            path = path[len(prefix):] or "/"
            break

    segments = [
        "{id}" if _is_identifier(seg) else seg
        for seg in path.split("/")
    ]
    return "/".join(segments) or "/"


class CountingTransport(httpx.AsyncBaseTransport):
    """Wraps a transport, recording one ``api_usage`` row per request.

    Recording is awaited inline rather than fired off as a task: the write is a
    single insert into a WAL-mode SQLite file, and doing it here keeps the row
    ordered against the response that produced it. It cannot fail the request —
    :func:`~backend.app.services.api_usage.record_api_usage` swallows everything.
    """

    def __init__(
        self,
        inner: httpx.AsyncBaseTransport,
        *,
        service: str,
        user_id: str | None = None,
    ) -> None:
        self._inner = inner
        self._service = service
        self._user_id = user_id

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        endpoint = normalise_endpoint(request.url)
        started = time.perf_counter()
        try:
            response = await self._inner.handle_async_request(request)
        except Exception:
            # No response was produced, so there is no status and no rate-limit
            # reading — but the request was still made, and a backfill that dies
            # on DNS is exactly the kind of thing this table should show.
            await record_api_usage(
                service=self._service,
                endpoint=endpoint,
                method=request.method,
                status_code=None,
                duration_ms=_elapsed_ms(started),
                user_id=self._user_id,
            )
            raise

        await record_api_usage(
            service=self._service,
            endpoint=endpoint,
            method=request.method,
            status_code=response.status_code,
            duration_ms=_elapsed_ms(started),
            user_id=self._user_id,
            rate_limit=parse_rate_limit(response.headers),
        )
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


def _elapsed_ms(started: float) -> int:
    """Milliseconds since *started*.

    Measured to response *headers*, not to the last byte of the body: that is
    where the transport hands back, and it is the figure that reflects the
    provider rather than the size of what we asked for.
    """
    return int((time.perf_counter() - started) * 1000)


def provider_client(
    service: str,
    *,
    timeout: Optional[httpx.Timeout] = None,
    follow_redirects: bool = False,
    user_id: str | None = None,
    **kwargs,
) -> httpx.AsyncClient:
    """An ``httpx.AsyncClient`` that counts every request it makes.

    The drop-in replacement for ``httpx.AsyncClient(...)`` at every provider call
    site. ``user_id`` is optional; when omitted the recorded row takes whatever
    :func:`~backend.app.services.api_usage.attribute_to_user` has established for
    the surrounding context, and NULL when nothing has.
    """
    return httpx.AsyncClient(
        transport=CountingTransport(
            httpx.AsyncHTTPTransport(), service=service, user_id=user_id
        ),
        timeout=timeout,
        follow_redirects=follow_redirects,
        **kwargs,
    )
